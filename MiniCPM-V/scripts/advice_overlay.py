# -*- coding: utf-8 -*-
"""advice_overlay.py —— 把「窗户定位结果」画进 gcs_gate 的识别画面（8091 那张图）

设计原则（照 gcs_gate.draw_hud 的风格）
  · **纯显示，异常自己吃掉** —— 画失败也绝不影响检测主循环
  · **只加不改** —— 在 gcs_gate.py 里只插一行调用
  · **读不到就画空** —— 没有建议时显示 "VLM --"，不抛异常
  · **坐标归一化** —— advice.json 里的位置是 0~1，网页图分辨率改了也不用动代码

advice.json 约定（由 fly_console.py 写）
  {
    "req_ts": 1786045000.2,      # 最近一次"按 Y"的时间戳（单调递增即可）
    "state": "ok" | "fail" | "pending",
    "text": "窗户 · 32 m · 约 1 楼（窗上有红点）",
    "source": "dot+gdino",
    "box": [x1,y1,x2,y2],        # 归一化 0~1
    "center": [cx,cy],           # 归一化
    "dot": [u,v],                # 归一化；没有就 null
    "dist_m": 31.8,
    "height_m": 0.82,
    "floor": "约 1 楼",
    "ms": {"dot":37,"seg":434,"gdino":2653,"total":3249}
  }

插到 gcs_gate.py（就这一行，位置在 `pv_publish(vis, meta, jpg)` 之前）：

    from advice_overlay import draw_advice
    vis = draw_advice(vis)          # ← 新增

自测
    python scripts\\advice_overlay.py            # 用 frame 254 生成一张对照图
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# ── 配置（可用环境变量覆盖）────────────────────────────────────────
def _default_advice_path():
    """建议 JSON 的默认路径。

    优先级: CPMV_ADVICE → $TEMP / $TMP → tempfile.gettempdir()
    为什么不是直接 tempfile.gettempdir(): tempfile 在**临时目录不可写**时
    会一路退到 os.getcwd()（实测过: 沙箱/受限环境下 gettempdir() 返回当前目录），
    那样写端和读端就可能落到不同地方。先吃 $TEMP 可以保证 Windows 上行为
    和之前完全一致，Linux 上没有 TEMP 才走 gettempdir()（= /tmp）。
    """
    p = os.environ.get("CPMV_ADVICE")
    if p:
        return Path(p)
    for k in ("TEMP", "TMP", "TMPDIR"):
        d = os.environ.get(k)
        if d and os.path.isdir(d):
            return Path(d) / "cpmv_advice.json"
    return Path(tempfile.gettempdir()) / "cpmv_advice.json"


ADVICE_PATH = _default_advice_path()
MAX_AGE_S = 600.0          # 超过这么久没更新 → 不再显示（防止显示上次的旧结果）

_state = {"mtime": None, "key": None, "advice": None, "t": 0.0, "err": None}

_FONTS = {}

# 中文字体候选：Windows / Linux(WQY、Noto) / macOS
# 也可以用环境变量 CPMV_FONT 指定一个 .ttf/.ttc
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


def _font(size):
    """找得到中文字体就返回 ImageFont；找不到返回 None（调用方退 cv2.putText）。"""
    if size in _FONTS:
        return _FONTS[size]
    f = None
    try:
        from PIL import ImageFont
    except Exception:
        ImageFont = None
    if ImageFont is not None:
        cands = ([os.environ["CPMV_FONT"]] if os.environ.get("CPMV_FONT") else []) + list(_FONT_CANDIDATES)
        for p in cands:
            if not p or not os.path.exists(p):
                continue
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                pass
    _FONTS[size] = f
    return f


def _put_ascii(bgr, items):
    """PIL 或中文字体都没有时的兜底：cv2.putText 画 ASCII（中文字符被过滤）。"""
    for text, xy, size, col in items:
        s = "".join(ch for ch in text if 32 <= ord(ch) < 127) or "-"
        # items 的 xy 按 PIL 左上角约定，cv2.putText 按左下角基线 → y 下移 size
        cv2.putText(bgr, s, (xy[0], xy[1] + size),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, size / 24.0), col, 2)
    return bgr


def _put_cn(bgr, items):
    """items=[(text,(x,y),size,(B,G,R))]，一次 PIL 往返画完；PIL 或中文字体不可用就退 ASCII。"""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return _put_ascii(bgr, items)
    if any(_font(size) is None for _, _, size, _ in items):
        return _put_ascii(bgr, items)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for text, xy, size, col in items:
        f = _font(size)
        if f is None:
            continue
        d.text(xy, text, font=f, fill=(col[2], col[1], col[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════
def load_advice(path=None, max_age=MAX_AGE_S):
    """mtime 变了才解析（每帧调用只有一次 os.stat 的开销）。异常一律返回 None。"""
    p = Path(path or ADVICE_PATH)
    try:
        st = p.stat()
    except Exception:
        _state["advice"] = None
        return None
    # 缓存键要带上路径: 否则用 --advice-file 换文件时, 只要 mtime 撞上就会读到上一个文件的缓存
    key = (str(p), st.st_mtime)
    if _state.get("key") != key:
        _state["key"] = key
        _state["mtime"] = st.st_mtime
        try:
            _state["advice"] = json.loads(p.read_text(encoding="utf-8"))
            _state["t"] = time.time()
            _state["err"] = None
        except Exception as e:
            _state["advice"] = None
            _state["err"] = "%s: %s" % (type(e).__name__, e)
    a = _state["advice"]
    if a is None:
        return None
    # 过期判据用**文件自身的 mtime**, 不是「我上次解析它的时刻」
    #   —— 否则写端 (bridge / fly_console) 挂掉后, 最后一条建议还会在画面上再留 max_age (默认 10 分钟)
    if time.time() - st.st_mtime > max_age:
        return None
    return a


def _px(v, n, fallback=0.0):
    try:
        return int(round(float(v) * n))
    except Exception:
        return int(round(fallback * n))


def draw_advice(bgr, path=None, max_age=MAX_AGE_S):
    """把建议画到 bgr（原地返回新图）。永远不抛异常。

    **没有建议文件 / 已过期 → 原样返回 bgr**（不画黑条、不做 PIL 转换）。
    """
    try:
        a = load_advice(path, max_age)
        if not a:
            return bgr                      # ← 没东西可画就什么都不做
        H, W = bgr.shape[:2]
        out = bgr
        if a.get("state") != "ok":
            txt = ("VLM: 看不清，请手动挪一下" if a.get("state") == "fail" else "VLM: 正在看…")
            return _bar(out, [(txt, (0, 200, 255))], None)
        # 框
        if a.get("box"):
            x1, y1, x2, y2 = a["box"]
            p1 = (_px(x1, W), _px(y1, H))
            p2 = (_px(x2, W), _px(y2, H))
            cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
            # 四角加粗，便于远看
            L = max(10, (p2[0] - p1[0]) // 6)
            for (cx, cy, dx, dy) in ((p1[0], p1[1], 1, 1), (p2[0], p1[1], -1, 1),
                                     (p1[0], p2[1], 1, -1), (p2[0], p2[1], -1, -1)):
                cv2.line(out, (cx, cy), (cx + dx * L, cy), (0, 255, 0), 3)
                cv2.line(out, (cx, cy), (cx, cy + dy * L), (0, 255, 0), 3)
        # 中心十字
        if a.get("center"):
            cx, cy = _px(a["center"][0], W), _px(a["center"][1], H)
            cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        # 红点
        if a.get("dot"):
            u, v = _px(a["dot"][0], W), _px(a["dot"][1], H)
            cv2.circle(out, (u, v), max(8, W // 200), (0, 0, 255), 2)
        lines = [("VLM · " + (a.get("text") or ""), (0, 255, 0))]
        src = a.get("source") or "-"
        ms = a.get("ms") or {}
        sub = "src %s   %s ms   (dot %s / seg %s / gdino %s)" % (
            src, ms.get("total", "-"), ms.get("dot", "-"), ms.get("seg", "-"), ms.get("gdino", "-"))
        lines.append((sub, (170, 170, 170)))
        out = _bar(out, lines, a)
        # ── 左上角信息牌：闸门 / 来源分数（2026-11-24 加；靠边内缩 8px，避开他的整幅黄边框）──
        try:
            out = _badge(out, a)
        except Exception:
            pass
        return out
    except Exception as e:
        try:
            print("[advice_overlay] 画建议异常(忽略): %s: %s" % (type(e).__name__, e), flush=True)
        except Exception:
            pass
        return bgr


def _badge(img, advice):
    """左上角小信息牌：闸门结果 + 窗框来源/分数。

    位置：左边内缩 8px、上边内缩 8px —— 避开他 canvas 画的「整幅一圈 2px 黄框」。
    不会和他的候选序号撞：序号画在候选点右上，而候选点一般在画面中部。
    """
    g = advice.get("gate") or {}
    w = advice.get("win") or {}
    if not g and not w:
        return img
    items = []
    if g:
        ok = bool(g.get("passed"))
        txt = "nw=%s  VLM=%s  %s" % (g.get("nw", "-"), g.get("vlm") or "-",
                                     "通过" if ok else "可疑")
        items.append((txt, (0, 255, 0) if ok else (0, 90, 255)))
    if w:
        try:
            items.append(("窗 %s %.2f  %d×%dpx" % (w.get("src", "-"), float(w.get("score") or 0),
                                                  int(w.get("w_px") or 0), int(w.get("h_px") or 0)),
                          (0, 255, 0)))
        except Exception:
            pass
    if not items:
        return img
    H, W = img.shape[:2]
    size = max(13, int(H * 0.021))
    lh = int(size * 1.5)
    pad = 6
    tw = max(int(len(t) * size * 0.62) for t, _ in items)     # 粗估宽度（中文按 0.62 倍字号）
    bw = min(W - 16, tw + pad * 2)
    bh = lh * len(items) + pad
    x0, y0 = 8, 8
    plate = img[y0:y0 + bh, x0:x0 + bw].copy()
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
    img[y0:y0 + bh, x0:x0 + bw] = cv2.addWeighted(img[y0:y0 + bh, x0:x0 + bw], 0.65, plate, 0.35, 0)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (90, 90, 90), 1)
    put = []
    y = y0 + pad // 2
    for t, col in items:
        put.append((t, (x0 + pad, y), size, col))
        y += lh
    return _put_cn(img, put)


def _bar(img, lines, advice):
    """底部信息条。lines=[(text,(B,G,R)), ...]"""
    H, W = img.shape[:2]
    bh = max(34, int(H * 0.062)) * len(lines)
    cv2.rectangle(img, (0, H - bh), (W, H), (0, 0, 0), -1)
    cv2.line(img, (0, H - bh), (W, H - bh), (0, 255, 0), 1)
    size = max(12, int(H * 0.026))
    items = []
    y = H - bh + int(size * 0.35)
    for txt, col in lines:
        items.append((txt, (10, y), size, col))
        y += int(size * 1.45)
    return _put_cn(img, items)


def to_advice_dict(r, W, H):
    """WindowFinder.find() 的结果 → advice.json（坐标归一化）。"""
    def n(v, n_):
        return round(float(v) / float(n_), 4)

    d = {"req_ts": time.time(), "state": "ok" if r.get("ok") else "fail",
         "text": r.get("text") or "", "source": r.get("source") or "none",
         "dist_m": r.get("dist_m"), "height_m": r.get("height_m"),
         "floor": r.get("floor"), "ms": r.get("ms") or {}}
    if r.get("box"):
        x1, y1, x2, y2 = r["box"]
        d["box"] = [n(x1, W), n(y1, H), n(x2, W), n(y2, H)]
    if r.get("center"):
        d["center"] = [n(r["center"][0], W), n(r["center"][1], H)]
    if r.get("dot"):
        d["dot"] = [n(r["dot"]["u"], W), n(r["dot"]["v"], H)]
    else:
        d["dot"] = None
    if r.get("gate"):
        g = r["gate"]
        d["gate"] = {"nw": g.get("nw"), "vlm": g.get("vlm"), "passed": g.get("passed"),
                     "why": g.get("why")}
    if r.get("win"):
        d["win"] = r["win"]
    return d


def write_advice(d, path=None):
    p = Path(path or ADVICE_PATH)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except Exception as e:
        print("[advice_overlay] 写 advice 失败: %s" % e, flush=True)
        return False


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "eval"))
    import bench_lat_crop as L
    from mcpv.window_finder import WindowFinder

    fr = L.load_frames()
    bgr = fr[254]["clean"]
    H, W = bgr.shape[:2]
    with WindowFinder() as wf:
        wf.wait_ready(300)
        r = wf.find(bgr)
    d = to_advice_dict(r, W, H)
    tmp = ROOT / "eval" / "advice_demo.json"
    write_advice(d, tmp)
    out = draw_advice(bgr, path=tmp)
    p = ROOT / "eval" / "figures" / "overlay_demo.jpg"
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        Path(p).write_bytes(buf.tobytes())
    print(json.dumps(d, ensure_ascii=False, indent=1))
    print("saved", p)
