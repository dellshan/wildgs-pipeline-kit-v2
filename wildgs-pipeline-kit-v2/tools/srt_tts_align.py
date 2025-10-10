#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, asyncio, os, re, tempfile, shutil
from typing import List, Tuple
from pydub import AudioSegment
import edge_tts

TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)

def hmsms_to_ms(h: int, m: int, s: int, ms: int) -> int:
    return ((h * 60 + m) * 60 + s) * 1000 + ms

def clean_text(s: str) -> str:
    # 去掉常见样式/标签，换行合并为空格
    s = s.replace("\u202a", "").replace("\u202c", "")
    s = re.sub(r"<[^>]+>", "", s)           # HTML/ASS 标签
    s = re.sub(r"\{\\.*?\}", "", s)         # ASS 样式
    s = s.replace("\\N", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_srt(path: str) -> List[Tuple[int, int, str]]:
    """返回 [(start_ms, end_ms, text), ...]，按时间排序"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 用空行分块，兼容不同换行
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())
    entries: List[Tuple[int, int, str]] = []

    for blk in blocks:
        lines = [ln for ln in blk.splitlines() if ln.strip() != ""]
        if not lines:
            continue

        # 常见结构：index 行 + 时间行 + 文本行
        # 也兼容无 index 的情况
        m = None
        time_line_idx = 0
        if re.match(r"^\d+$", lines[0].strip()):
            if len(lines) >= 2:
                m = TIME_RE.search(lines[1])
                time_line_idx = 1
        else:
            m = TIME_RE.search(lines[0])
            time_line_idx = 0

        if not m:
            # 跳过异常块
            continue

        sh, sm, ss, sms, eh, em, es, ems = map(int, m.groups())
        start_ms = hmsms_to_ms(sh, sm, ss, sms)
        end_ms   = hmsms_to_ms(eh, em, es, ems)
        if end_ms <= start_ms:
            # 保底：至少给 100ms
            end_ms = start_ms + 100

        text_lines = lines[time_line_idx + 1:]
        text = clean_text("\n".join(text_lines))
        if not text:
            # 没有文本也保留时轴（可以用静音占位）
            text = ""

        entries.append((start_ms, end_ms, text))

    # 按时间排序
    entries.sort(key=lambda x: x[0])
    return entries

async def tts_to_file(text: str, voice: str, rate: str, volume: str, out_path: str):
    """edge-tts：把 text 合成为 out_path（mp3）"""
    # 空文本也生成极短静音（pydub 来做）；这里直接跳过，让调用方补静音即可
    if not text.strip():
        # 写一个 10ms 的空白 mp3（为了流程统一）
        silent = AudioSegment.silent(duration=10)
        silent.export(out_path, format="mp3", bitrate="192k")
        return
    comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    await comm.save(out_path)

def build_timeline(
    entries: List[Tuple[int, int, str]],
    tmp_dir: str,
    voice: str,
    rate: str,
    volume: str,
    sr: int,
    mode: str = "cut",   # "cut": 超时截断；"pad": 允许溢出拼接至下一句（不建议）
    verbose: bool = True,
) -> AudioSegment:
    """
    用 pydub 把每句音频放到对应时间段里：
      - 如果一句 TTS 时长 < 槽位：末尾补静音到 end_ms
      - 如果 > 槽位：'cut' 直接硬截；（'pad' 会溢出到后面，不严格对齐）
    返回整条时间线上对齐好的 AudioSegment
    """
    timeline = AudioSegment.silent(duration=0, frame_rate=sr).set_frame_rate(sr)
    cursor = 0

    loop = asyncio.get_event_loop()
    for i, (st, ed, txt) in enumerate(entries, 1):
        slot = max(10, ed - st)
        # 先生成该句 TTS 临时文件
        tmp_mp3 = os.path.join(tmp_dir, f"seg_{i:05d}.mp3")
        if verbose:
            print(f"[TTS {i}/{len(entries)}] {st}→{ed}ms  {txt[:30]}{'...' if len(txt)>30 else ''}")
        loop.run_until_complete(tts_to_file(txt, voice, rate, volume, tmp_mp3))

        seg = AudioSegment.from_file(tmp_mp3)
        seg = seg.set_frame_rate(sr).set_channels(1)

        # 把时间线补到 start
        if st > cursor:
            timeline += AudioSegment.silent(duration=st - cursor, frame_rate=sr)

        if len(seg) <= slot:
            # 不足则补静音
            seg = seg + AudioSegment.silent(duration=slot - len(seg), frame_rate=sr)
            timeline += seg
        else:
            if mode == "cut":
                # 超过则硬截，尾部可加轻微淡出避免咔啪
                seg = seg[:slot].fade_out(min(40, slot//8))
                timeline += seg
            else:
                # pad 模式：直接拼接（不再保证严格对齐）
                timeline += seg

        cursor = st + slot

    return timeline.set_frame_rate(sr).set_channels(1)

def main():
    parser = argparse.ArgumentParser(description="按 SRT 时间轴精确对齐的中文配音合成（edge-tts + pydub）")
    parser.add_argument("--srt", required=True, help="输入 SRT 文件路径")
    parser.add_argument("--out", default="zh_voice.mp3", help="输出 mp3 文件（默认 zh_voice.mp3）")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural", help="TTS 发音人（edge-tts 语音名）")
    parser.add_argument("--rate", default="-10%", help="语速，例：0%、-10%、+10%")
    parser.add_argument("--volume", default="+0%", help="音量，例：+0%、+5%")
    parser.add_argument("--samplerate", type=int, default=48000, help="输出采样率（默认 48000）")
    parser.add_argument("--mode", choices=["cut", "pad"], default="cut",
                        help="超出时长如何处理：cut=截断以严格对齐；pad=不截断（不严格对齐）")
    parser.add_argument("--bitrate", default="192k", help="mp3 比特率（默认 192k）")
    parser.add_argument("--no_progress", action="store_true", help="不打印进度")
    args = parser.parse_args()

    entries = parse_srt(args.srt)
    if not entries:
        raise SystemExit(f"[ERR] 解析不到字幕条目：{args.srt}")

    tmp_dir = tempfile.mkdtemp(prefix="srt_tts_")
    try:
        timeline = build_timeline(
            entries=entries,
            tmp_dir=tmp_dir,
            voice=args.voice,
            rate=args.rate,
            volume=args.volume,
            sr=args.samplerate,
            mode=args.mode,
            verbose=not args.no_progress,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        timeline.export(args.out, format="mp3", bitrate=args.bitrate)
        print(f"[OK] 导出：{args.out}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
