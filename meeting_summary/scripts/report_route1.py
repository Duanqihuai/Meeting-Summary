"""汇总真实路线一输出，生成中文说明与可点击证据的离线网页。

只读取已完成结果，不调用API，不修改模型摘要；参考标注只用于事后诊断。
"""
import argparse
from collections import Counter
import csv
import html
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from common.core import read_json, write_json, check_evidence
from common.diarization import union_duration


def speaker_diagnostic(reference, turns):
    """参考词中点上的单人区域诊断；不是标准DER，不混入模型输入。"""
    refs = sorted({w["speaker"] for w in reference})
    preds = sorted({t["speaker"] for t in turns})
    confusion = Counter()
    eligible = available = 0
    for word in reference:
        t = (word["start"] + word["end"]) / 2
        active_ref = {w["speaker"] for w in reference if w["start"] <= t < w["end"]}
        if active_ref != {word["speaker"]}:
            continue
        eligible += 1
        active_pred = {turn["speaker"] for turn in turns if turn["start"] <= t < turn["end"]}
        if len(active_pred) == 1:
            available += 1
            confusion[(next(iter(active_pred)), word["speaker"])] += 1
    if len(preds) > 8:
        return {"status": "skipped", "reason": "簇数过多，未运行排列匹配"}
    candidates = refs + [f"UNMATCHED_{i}" for i in range(max(0, len(preds) - len(refs)))]
    best, mapping = -1, {}
    for perm in itertools.permutations(candidates, len(preds)):
        value = sum(confusion[(p, r)] for p, r in zip(preds, perm))
        if value > best:
            best, mapping = value, dict(zip(preds, perm))
    return {"metric": "reference-word-midpoint single-speaker attribution diagnostic, not DER",
            "eligible_reference_words": eligible, "predicted_single_speaker_words": available,
            "correct_after_optimal_label_mapping": best,
            "coverage": available / eligible if eligible else None,
            "accuracy_including_unknown_as_wrong": best / eligible if eligible else None,
            "accuracy_when_single_prediction": best / available if available else None,
            "evaluation_only_label_mapping": mapping,
            "confusion": [{"prediction": a, "reference": b, "count": count} for (a, b), count in confusion.items()]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True, help="包含run子目录的独立实验文件夹")
    p.add_argument("--reference-root", help="可选：先前AMI原始参考导入文件夹（含reference_words.json）")
    args = p.parse_args()
    root = Path(args.result).resolve()
    run = root / "run"
    meta, summary, segments = [read_json(run / f"{name}.json") for name in ["run", "summary", "segments"]]
    if meta["status"] != "completed":
        raise ValueError("只汇总已完成流程")
    ids = {s["id"]: s for s in segments}
    check_evidence(summary, ids)
    turns, calls, asr_words = [read_json(run / f"{name}.json") for name in ["diarization", "raw_calls", "asr_words"]]
    source = Path(meta["source"])
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[1] / source
    if not (root / "meeting.wav").exists():
        shutil.copyfile(source, root / "meeting.wav")
    speakers = sorted({t["speaker"] for t in turns})
    speaker_rows = []
    for label in speakers:
        rows = [s for s in segments if s["speaker"] == label]
        seconds = union_duration([(s["start"], s["end"]) for s in turns if s["speaker"] == label])
        speaker_rows.append({"speaker": label, "model_speech_seconds": round(seconds, 2), "transcript_segments": len(rows)})
    write_json(root / "speaker_overview.json", speaker_rows)
    with (root / "speaker_overview.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(speaker_rows[0]))
        writer.writeheader()
        writer.writerows(speaker_rows)
    usage = {k: sum((c.get("usage") or {}).get(k, 0) or 0 for c in calls)
             for k in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    failed_calls = []
    for path in root.glob("attempt_*/raw_calls.json"):
        failed_calls.extend(read_json(path))
    failed_tokens = sum((c.get("usage") or {}).get("total_tokens", 0) or 0 for c in failed_calls)
    earlier_calls = []
    for path in root.glob("draft_*/raw_calls.json"):
        earlier_calls.extend(read_json(path))
    earlier_tokens = sum((c.get("usage") or {}).get("total_tokens", 0) or 0 for c in earlier_calls)
    write_json(root / "api_usage.json", {"calls": len(calls), "tokens": usage,
        "failed_attempt_calls": len(failed_calls), "failed_attempt_total_tokens": failed_tokens,
        "earlier_draft_calls": len(earlier_calls), "earlier_draft_tokens": earlier_tokens,
        "connectivity_probe_tokens": 44, "all_recorded_tokens_including_probe": usage["total_tokens"] + failed_tokens + earlier_tokens + 44,
        "note": "分别保留成功摘要、失败尝试与连通性测试；费用以账户账单为准"})
    diagnosis_text = "未提供参考标注，仅验证流程与引用格式。"
    if args.reference_root:
        import jiwer
        from scripts.evaluate import tokens
        reference_root = Path(args.reference_root)
        reference = read_json(reference_root / "reference_words.json")
        utterances = read_json(reference_root / "reference_utterances.json")
        scope = read_json(reference_root / "data_manifest.json")
        begin, end = scope["score_start"], scope["score_end"]
        ref = " ".join(s["text"] for s in utterances)
        hyp = " ".join(w["text"] for w in asr_words if begin <= (w["start"] + w["end"]) / 2 < end)
        result = jiwer.process_words(" ".join(tokens(ref, "word")), " ".join(tokens(hyp, "word")))
        evaluation = {"scope_start": begin, "scope_end": end, "wer": result.wer,
            "substitutions": result.substitutions, "deletions": result.deletions, "insertions": result.insertions,
            "note": "全录音推理后按词中点取标注范围，和此前裁剪后推理的WER不完全相同",
            "speaker_diagnostic": speaker_diagnostic(reference, turns)}
        write_json(root / "evaluation/metrics.json", evaluation)
        shutil.copyfile(reference_root / "reference_words.json", root / "evaluation/reference_words.json")
        (root / "evaluation/wer_alignment.txt").write_text(jiwer.visualize_alignment(result), encoding="utf-8")
        d = evaluation["speaker_diagnostic"]
        diagnosis_text = (f"事后对标注范围{begin:.2f}–{end:.2f}秒计分，ASR WER为{result.wer:.2%}。"
            "本次先识别整段录音，再按词中点截取计分；此前是先裁剪再识别，因此不要将差异归因于说话人模型。\n\n"
            f"在人工词中点上只选参考中单人活动的区域，共{d['eligible_reference_words']}个参考词；"
            f"模型在其中给出单一说话人的覆盖率为{d['coverage']:.2%}。允许匿名标签与参考A/B/C/D做最优一一对应后，"
            f"把未知或多人预测也算错，归属正确比例为{d['accuracy_including_unknown_as_wrong']:.2%}。"
            "这是词中点上的粗略诊断，不是标准DER；不衡量转写文字是否正确，也不代表责任人识别准确率。"
            "映射只用于评测，没有写回模型输出，也没有输入DeepSeek。参考标注范围外的头尾内容没有金标准，不评价准确率。")
    (root / "requirements-lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    table = "\n".join(f"|{r['speaker']}|{r['model_speech_seconds']:.2f}秒|{r['transcript_segments']}|" for r in speaker_rows)
    text = f'''# 路线一完整测试说明：自动区分说话人 → 中文会议摘要

## 已经实际完成什么

本次使用AMI ES2002a完整混音（{meta['audio_seconds']:.2f}秒，约{meta['audio_seconds']/60:.2f}分钟），重新执行本地small语音识别、说话人模型、逐词时间对齐、DeepSeek API分块摘要和归并。录音是英语，最终摘要是简体中文。不是人工演示样例，也没有把人工参考转写交给大模型。

最直观的入口是 [查看会议摘要.html](查看会议摘要.html)：先看中文纪要，点击任何证据按钮可跳转到原音频对应位置，并显示该段英文转写。纯文本结果见 [最终会议摘要.md](最终会议摘要.md)。

**阅读结论时请同时看 [会议摘要_审阅整理版.md](会议摘要_审阅整理版.md) 与 [效果审阅与局限.md](效果审阅与局限.md)。机器原版存在“10 to 12”译成“10点12分”的错误，ASR还有金额小数点错误。审阅版单独标明修改依据，不冒充机器原始输出。**

## 说话人是怎么来的

切分网络为pyannote segmentation 3.0 ONNX，声纹网络为NeMo TitaNet small ONNX，通过sherpa-onnx在CPU运行。用已知“4人参会”的信息约束聚类，实际得到{len(speakers)}个匿名簇。每一段属于哪个簇是模型判断的，没有读取人工说话人标注。

|匿名标签|模型发言区间时长（可重叠）|对应转写片段数|
|---|---:|---:|
{table}

标签仅在本录音有效，不能直接称为真实姓名或职位。不同人的发言区间可能同时存在，时长之和可能超过总时长。说话人模型不会分离出独立声音，也不能恢复ASR漏词。

ASR输出{meta['asr_word_count']}个词级条目，其中{meta['uncertain_speaker_words']}个（{meta['uncertain_speaker_word_ratio']:.2%}）由于没有足够时间覆盖或多个候选竞争而保留未知身份；重组后共{len(segments)}条证据片段。时间覆盖比例是启发式，不是模型校准概率，即使有标签也可能归错。

## 耗时与API调用

- 语音识别（含加载）：{meta['asr_seconds_including_load']:.2f}秒。
- 说话人流程（含模型加载）：{meta['diarization']['elapsed_seconds']:.2f}秒。
- 成功阶段总处理耗时：{meta['elapsed_seconds']:.2f}秒（语音阶段＋摘要恢复阶段，不含失败与调试等待）。这是阶段耗时合计，不能当作一次不中断执行的墙钟时间或稳定基准。
- 配置的文本模型：`{meta['text_model']}`；成功记录的文本调用{len(calls)}次。
- 最终摘要API报告输入{usage['prompt_tokens']}、输出{usage['completion_tokens']}、合计{usage['total_tokens']} token。初版摘要另有{earlier_tokens} token，失败调用{failed_tokens} token，连通性测试44 token，所有已记录调用共{usage['total_tokens'] + failed_tokens + earlier_tokens + 44} token；没有猜测人民币费用。

实际服务端模型名、请求ID、每次token用量保存在run/raw_calls.json。密钥从本机.env读取，未写入本结果文件夹。录音没有发送给DeepSeek，接口接收的是英文转写、匿名标签、时间和证据编号。

## 效果该怎样理解

{diagnosis_text}

所有决定、行动项和风险的证据编号都通过存在性检查，但这只能证明“引用了一个真实片段”，不能证明该片段支持该结论。负责人还可能受说话人聚类误差和ASR遗漏影响，不能跳过听音核验。机器摘要按模型原样保留；如有人工审阅意见，将单独存放，避免把人工修改伪装成模型效果。

## 文件与复现

- `run/summary.json`、`run/summary.md`：模型最终输出，含证据索引。
- `run/asr_words.json`、`run/asr_segments.json`：本地ASR输出。
- `run/diarization.json`：模型说话人时间区间。
- `run/speaker_words.json`、`run/segments.json`：逐词归属及重组片段。
- `run/speech_run.json`、`run/run.json`：模型哈希、人数设置、参数和耗时。
- `speaker_overview.csv`、`api_usage.json`：统计汇总。
- `evaluation/`：仅用于事后诊断的参考与计分，不进入识别/摘要模型。
- `meeting.wav`：完整混音副本，供网页离线回放。源数据署名AMI Meeting Corpus, AMI Consortium，许可CC BY 4.0，来源https://groups.inf.ed.ac.uk/ami/download/ 。

在meeting_summary目录执行：

```powershell
.\\.venv\\Scripts\\python.exe run.py --route 1 --audio data/raw/ami/ES2002a/ES2002a.Mix-Headset.wav --asr-model models/weights/faster-whisper-small --language en --num-speakers 4 --out ../Route1_test_new/run
```

输出目录必须是新的；生成展示文档使用 `python -m scripts.report_route1 --result ../Route1_test_new --reference-root ../ASR_tests_20261118/inputs`。使用参考评测需安装requirements-eval.txt。完整原理与中文使用说明位于项目docs/04_路线一说话人增强.md。自动化测试与界面配置检查记录见验证记录.md。

attempt_01_failed保留了第一次在JSON写入阶段失败的记录，该次没有调用摘要API。修复后重新执行ASR与说话人阶段；随后DeepSeek默认推理模式用完4096个输出token却无正文，保存在attempt_02_llm_failed。关闭推理模式后，通过 `--resume-summary` 校验原音频SHA256并复用已成功的语音结果，完成真实摘要调用。恢复没有使用人工转写，没有重复识别音频。`run/speech_run.json`和`run/run.json`分别记录语音与恢复摘要的信息。

初版分块摘要保存在draft_01中；检查发现其遗漏后文会议时间，并保留了前文“未明确”的判断。最终版本增加全文转写复核，核对后续安排、任务和数值冲突；全文复核仍是模型操作，不等于人工认证。归并和复核均只使用ASR转写，没有读取人工参考摘要。
'''
    (root / "测试说明.md").write_text(text, encoding="utf-8")
    # 从机器输出重建展示，不修改决定或行动项内容。
    md = ["# 最终会议摘要（DeepSeek实际生成）", "", summary["overview"], "",
          "> 原音频为英语；下文为模型生成的中文摘要。SPEAKER编号为匿名模型标签，责任归属需复核。", ""]
    html_parts = []
    for key, title in [("decisions", "主要决定"), ("actions", "行动项"), ("risks", "风险与待确认事项")]:
        md += ["## " + title, ""]
        cards = []
        for row in summary[key]:
            suffix = f"；负责人：{row.get('owner') or '待确认'}；期限：{row.get('deadline') or '未明确'}" if key == "actions" else ""
            md += [f"- {row['text']}{suffix}（证据：{', '.join(row['evidence_ids'])}）"]
            buttons = " ".join(f"<button class='evidence' data-id='{html.escape(sid, quote=True)}' data-start='{ids[sid]['start']}'>{html.escape(sid)} · {ids[sid]['start']:.1f}s</button>" for sid in row["evidence_ids"])
            cards.append(f"<article><p>{html.escape(row['text'] + suffix)}</p><div>{buttons}</div></article>")
        md.append("")
        html_parts.append(f"<h2>{title}</h2>" + "".join(cards))
    (root / "最终会议摘要.md").write_text("\n".join(md), encoding="utf-8")
    transcript = "".join(f"<p id='{html.escape(s['id'], quote=True)}'><button class='evidence' data-id='{html.escape(s['id'], quote=True)}' data-start='{s['start']}'>{s['start']:.1f}s</button> <b>{html.escape(s['speaker'] or '未知说话人')}</b> · {html.escape(s['id'])}<br>{html.escape(s['text'])}</p>" for s in segments)
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>会议摘要与说话人证据</title>
<style>body{max-width:1100px;margin:35px auto;padding:0 22px;font:16px/1.8 'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#263348}h1,h2{line-height:1.4}article,.intro{background:white;padding:18px 24px;margin:14px 0;border-radius:12px}button{border:1px solid #bdd3e2;background:#eaf3fa;border-radius:6px;padding:6px 10px;cursor:pointer;color:#23597e;margin:3px}audio{width:100%}.player{position:sticky;top:0;background:#f4f7fa;padding:10px 0;border-bottom:1px solid #ddd;z-index:1}#selected{padding:10px;max-height:160px;overflow:auto}details{background:white;padding:20px;margin-top:25px}details p{padding:12px;border-bottom:1px solid #eee}.focus{background:#fff3c5}a{color:#22649a}</style>
<h1>会议摘要 · 自动区分说话人</h1><p>AMI ES2002a · 英语录音 / 中文纪要 · 匿名标签须核对</p>
<section class='intro'><b>本页保留机器原始摘要。</b>审阅发现时间错译（10 to 12不等于10点12分）和ASR金额问题。请结合 <a href='会议摘要_审阅整理版.md'>审阅整理版</a> 与 <a href='效果审阅与局限.md'>问题说明</a> 阅读；下方原始内容未人工改写。</section>
<p><a href='测试说明.md'>测试说明</a> · <a href='最终会议摘要.md'>下载中文纪要</a> · <a href='run/summary.json'>查看模型JSON</a></p>
<div class='player'><audio id='audio' controls src='meeting.wav'></audio><div id='selected'>点击任何证据按钮，回放原音频并查看对应转写。</div></div>""" + f"<section class='intro'>{html.escape(summary['overview'])}</section>" + "".join(html_parts) + \
        "<details><summary>展开带说话人标签的完整转写（英文）</summary>" + transcript + "</details>" + """
<script>document.querySelectorAll('.evidence').forEach(button=>button.addEventListener('click',()=>{const audio=document.getElementById('audio');audio.currentTime=Number(button.dataset.start);audio.play().catch(()=>{});const row=document.getElementById(button.dataset.id);document.getElementById('selected').textContent=row?row.textContent:'未找到片段';}));</script></html>"""
    (root / "查看会议摘要.html").write_text(page, encoding="utf-8")
    print("报告与可回放摘要已生成：", root)


if __name__ == "__main__":
    main()
