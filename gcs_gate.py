#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gcs_gate.py — 地面站 格子环 + 红点 检测 (配飞机 vision_bridge _mode:=gate_gcs)
===============================================================================
飞机 → 这里:  1920×540 并排双目 JPEG (每目 960×540, 和机上一体版的工作分辨率相同) + t_cap
这里 → 飞机:  相机(校正左目)系观测 {ring, dots, diag} + 标定摘要 rig{f, R1, ...} + sim 标志
              **不需要也拿不到位姿**: 世界系融合 (估计器) 在飞机上做, 位姿在飞机上按 t_cap 插值。

★2026-10-16: 启动参数 / HUD / 控制台输出 对齐 gcs_aruco_det2.py
  用法 (和 gcs_aruco_det2 同一套参数):
         python gcs_gate.py --drone 192.168.2.20 --mode 1          (默认: 检测, sim, 飞机不会响应)
         python gcs_gate.py --drone 192.168.2.20 --mode 2          (★ 会驱动飞机)
         python gcs_gate.py --drone 192.168.2.20 --mode 0          (不检测, 只看画面, 照样回空包)
         python gcs_gate.py --drone 192.168.2.20 --probe           (强制 sim; 大字显示 环距/红点距)
  gate 版自己的:
         python gcs_gate.py --drone 192.168.2.20 --record rec/     (存上行原图, 离线调参用)
         python gcs_gate.py --replay rec/                          (不连飞机, 回放录下来的帧)
  老参数在 gate 版不起作用的 (传了只打一行提示, 不报错):
         --pitch / --roll        外参在飞机上 → 起桥时给 _cam_pitch_deg:= / _cam_roll_deg:=
         --with-sub              只对 ArUco 板有效
         --show-color / --preview-gray   画面固定是校正后左目彩色图 (= 检测器看到的)
  --real (旧写法) = --mode 2, 隐藏参数, 已有脚本不用改。

  同目录需要: vb09/ (common + perception)  stereo_calib.yaml  (gate_params.json 可选)
              网页「识别画面」用 vb09/common/preview_server.py

保险 (单保险, 和 gcs_aruco_det2 一样只看地面 --mode)
-------------------------------------------------------------------------------
  地面: --mode 1 (默认) / --probe → 回 sim=True; --mode 2 → sim=False (★ 驱动飞机)
  飞机: gate_gcs 模式下 _gate_sim 默认跟随地面; 显式给 _gate_sim:=true 可以在飞机上强制锁住。

显示 (同 gcs_aruco_det2)
-------------------------------------------------------------------------------
  本地窗口: 校正后左目 (带椭圆/角点/红点) 放大到 1920×1080, 顶部半透明黑带 HUD (8 行, 位置同老版)。
            HUD 标签用英文 —— OpenCV 自带字体画不了中文 (老版中文一直显示成 ???)。
  网页预览: 图上**不写字** (只有检测标记), 数字放在 X-Det 里由浏览器渲染。
            地址 http://127.0.0.1:8091/snapshot.jpg; 浏览器在另一台电脑: --preview-host 0.0.0.0
            X-Det kind="gate": 阶段 / 距离 / 来源 / 标准差 来自**飞机回传的融合结果**,
            环 / 角点 / 红点 / 检测耗时 来自本机检测。

和 gcs_aruco_det2 的关系: 同一套 ZMQ 端口 (5555 上行 / 5556 下行), 二选一运行。
  飞机桥是 aruco 模式时, 这里会提示"收到的不是并排帧"并只显示画面, 不回观测。
===============================================================================
"""
import argparse
import glob
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vb09.common import geom   # noqa: E402
from vb09.common import wire   # noqa: E402
from vb09.common.wire import pack, unpack, require_msgpack, zmq, rig_to_msg   # noqa: E402
from vb09.perception import registry   # noqa: E402
from vb09.perception.stereo_rig import StereoRig   # noqa: E402
# HUD / 掩膜视图 / 控制台诊断: 检测器有显示钩子 (hud_rows / ground_draw / console / mask / found / probe_dist,
#   例: vb09/perception/reddot/plugin.py) 就用钩子; 没有 (环) → 下面这些环专用的代码, 行为和以前一样 (2026-11-05)
from vb09.perception.ring.det2d import apply_wb, segment   # noqa: E402
from vb09.perception.ring.geom3d import ring_tilt_deg   # noqa: E402

require_msgpack("必须安装 msgpack: pip install msgpack")       # zmq 可以没有 (--replay 离线回放)

# =============================================================================
# 可选：网页「识别画面」底部叠加「第二意见」信息条 (2026-11-24 黄圣鑫)
#   · **默认关闭**, 用 --advice 显式打开; 不加这个参数时代码路径和以前完全一样
#   · 检测 / 推流 / 状态机 / 接口: 一行没动
#   · 画图在 advice_overlay.py, 查找顺序:
#       环境变量 CPMV_ADV_DIR → 与 gcs_gate.py 同目录 → <仓库根>/MiniCPM-V/scripts
#       → <上两级>/MiniCPM-V/scripts (本地开发布局)
#   · 找不到 / 加载失败 / 画挂了 → 自动跳过, 画面和以前一样, 不影响检测主循环
# =============================================================================
def _load_advice_overlay():
    """按需加载 advice_overlay.draw_advice；仅在 --advice / --advice-file 启用时调用。"""
    cands = []
    if os.environ.get("CPMV_ADV_DIR"):
        cands.append(os.environ["CPMV_ADV_DIR"])
    cands.append(HERE)
    cands.append(os.path.join(HERE, "MiniCPM-V", "scripts"))
    cands.append(os.path.normpath(os.path.join(HERE, "..", "..", "MiniCPM-V", "scripts")))
    adv_dir = next((d for d in cands
                    if d and os.path.exists(os.path.join(d, "advice_overlay.py"))), None)
    if not adv_dir:
        return None, None
    try:
        if adv_dir not in sys.path:
            sys.path.insert(0, adv_dir)
        from advice_overlay import draw_advice  # noqa: E402
        return draw_advice, adv_dir
    except Exception:
        return None, adv_dir

UP_PORT, DOWN_PORT = 5555, 5556
MODE_NAMES = {0: "OFF", 1: "GATE-SIM", 2: "GATE-REAL"}

# ---- HUD (坐标/字号/颜色同 gcs_aruco_det2 的 draw_hud) ----
OUT_W, OUT_H = 1920, 1080           # 本地窗口画布: 960×540 校正左目放大 2 倍
HUD_H = 224                         # 顶部黑带高度; 改了行数要同步改
HUD_Z_DEADBAND = 0.05               # 同 gcs_aruco_det2 / 飞机侧 TRACK_Z_DEADBAND, 只影响颜色

# 飞控模式 → (含义, 颜色 BGR)
_MODE_INFO = {
    "OFFBOARD":     ("program control (vision/auto)", (80, 255, 80)),
    "POSCTL":       ("manual: position",              (0, 200, 255)),
    "ALTCTL":       ("manual: altitude",              (0, 200, 255)),
    "STABILIZE":    ("manual: stabilize",             (0, 200, 255)),
    "MANUAL":       ("manual (raw)",                  (0, 200, 255)),
    "ACRO":         ("manual: acro",                  (0, 140, 255)),
    "AUTO.TAKEOFF": ("auto takeoff",                  (80, 220, 80)),
    "AUTO.LOITER":  ("auto hold",                     (80, 220, 80)),
    "AUTO.MISSION": ("auto mission",                  (80, 220, 80)),
    "AUTO.RTL":     ("* AUTO RETURN *",               (0, 120, 255)),
    "AUTO.LAND":    ("* AUTO LAND *",                 (0, 120, 255)),
}

# 飞行脚本阶段 → (含义, 颜色)
_PHASE_INFO = {
    "init":            ("init",                  (160, 160, 160)),
    "prime":           ("pre-takeoff prep",      (0, 200, 255)),
    "offboard_switch": ("switching OFFBOARD",    (0, 200, 255)),
    "arm":             ("arming",                (0, 200, 255)),
    "takeoff":         ("climbing",              (0, 255, 255)),
    "hover":           ("hover",                 (80, 220, 80)),
    "teleop":          ("* keyboard teleop *",   (0, 165, 255)),
    "track":           ("** VISION TRACKING **", (80, 255, 80)),
    "return_home":     ("returning home",        (255, 200, 0)),
    "descend":         ("descending",            (255, 200, 0)),
    "auto_land":       ("* auto land *",         (0, 120, 255)),
    "cleanup":         ("cleanup / motors off",  (160, 160, 160)),
}


# =============================================================================
# 收图: 专职线程读空 socket, 只留最新一帧 (同 gcs_aruco_det2 的 FrameSlot)
# =============================================================================
class FrameSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._raw = None
        self._ev = threading.Event()
        self.n_recv = 0
        self.n_skip = 0

    def put(self, raw):
        with self._lock:
            if self._raw is not None:
                self.n_skip += 1
            self._raw = raw
            self.n_recv += 1
        self._ev.set()

    def get(self, timeout=0.5):
        if not self._ev.wait(timeout):
            return None
        with self._lock:
            raw, self._raw = self._raw, None
            self._ev.clear()
        return raw


def reader_loop(sub, slot, stop):
    while not stop.is_set():
        try:
            raw = sub.recv()
        except zmq.Again:
            continue
        except Exception as e:
            print(f"[gcs_gate] 收图线程异常: {e}")
            time.sleep(0.2)
            continue
        slot.put(raw)
        while True:
            try:
                slot.put(sub.recv(zmq.NOBLOCK))
            except zmq.Again:
                break


def replay_loop(folder, slot, stop, hz):
    """--replay: 把 --record 存下的 *.msgpack 按 hz 循环喂进来 (t_cap 换成现在, 好看延迟)。"""
    files = sorted(glob.glob(os.path.join(folder, "*.msgpack")))
    if not files:
        print(f"[gcs_gate] {folder} 里没有 *.msgpack")
        stop.set()
        return
    print(f"[gcs_gate] 回放 {len(files)} 帧 @{hz}Hz (循环)")
    seq = 0
    while not stop.is_set():
        for f in files:
            if stop.is_set():
                break
            with open(f, "rb") as fh:
                head = unpack(fh.read())
            seq += 1
            head["seq"], head["t_cap"] = seq, time.time()
            if isinstance(head.get("gate"), dict):
                head["gate"]["t"] = head["t_cap"]       # 回放时融合结果的时间也换成现在, 不显示假滞后
            slot.put(pack(head))
            time.sleep(1.0 / hz)


# =============================================================================
# 统计 (照搬 gcs_aruco_det2 的 Stats; pnp=ok/tot 换成 环/红点 检出数)
# =============================================================================
class Stats:
    """
    往返 = 飞机端量的真·往返 (拍摄 → 结果回到飞机, 纯飞机时钟), 桥放在上行包头 rtt_* 里。
    上行≈ = 往返 − 本机检测耗时。旧桥没有 rtt_* 时退回"相对量"估计 (基准 = 历史最小)。
    """
    STALL_MS = 800.0

    def __init__(self):
        self.t0 = time.time()
        self.n = 0
        self.dropped = 0
        self.bytes = 0
        self.last_seq = 0
        self.gaps = 0
        self.dt = deque(maxlen=60)
        self._t_prev = None
        self.t_cap_off = None
        self.lat_ms = float('nan')
        self.lat_max = 0.0
        self.stalls = 0
        self._in_stall = False
        self.slot = None
        self.rtt = float('nan')
        self.rtt_p50 = float('nan')
        self.rtt_p95 = float('nan')
        self.rtt_max = 0.0
        self.rtt_n = 0
        self.rtt_stall = 0
        self.has_rtt = False
        self.det_ms = float('nan')
        self._det_dq = deque(maxlen=60)
        self.n_det = 0              # 检测过的帧
        self.n_ring = 0             # 其中环定位成功
        self.n_dot = 0              # 其中红点检出

    def tick(self, seq, nbytes, t_cap=None, head=None):
        now = time.time()
        self.n += 1
        self.bytes += nbytes
        if self._t_prev is not None:
            self.dt.append(now - self._t_prev)
        self._t_prev = now
        if self.last_seq and seq > self.last_seq + 1:
            self.gaps += seq - self.last_seq - 1
        self.last_seq = seq
        if t_cap:
            raw = (now - float(t_cap)) * 1000.0
            if self.t_cap_off is None or raw < self.t_cap_off:
                self.t_cap_off = raw
            lat = max(0.0, raw - self.t_cap_off)
            self.lat_ms = lat if self.lat_ms != self.lat_ms else 0.7 * self.lat_ms + 0.3 * lat
            if lat > self.lat_max:
                self.lat_max = lat
            if lat > self.STALL_MS:
                if not self._in_stall:
                    self.stalls += 1
                    self._in_stall = True
            else:
                self._in_stall = False
        if head is not None:
            try:
                _r = head.get("rtt_p50")
                if _r is not None and _r == _r:
                    _p50 = float(_r)
                    _t = head.get("rtt_p95")
                    _p95 = float(_t) if (_t is not None and _t == _t) else _p50
                    _l = head.get("rtt_ms")
                    _last = float(_l) if (_l is not None and _l == _l) else _p50
                    _m = head.get("rtt_max")
                    _mx = float(_m) if (_m is not None and _m == _m) else 0.0
                    self.rtt_p50, self.rtt_p95 = _p50, _p95
                    self.rtt, self.rtt_max = _last, _mx
                    self.rtt_n = int(head.get("rtt_n") or 0)
                    self.rtt_stall = int(head.get("rtt_stall") or 0)
                    self.has_rtt = True
            except (TypeError, ValueError):
                self.has_rtt = False

    def note_det_ms(self, ms):
        if ms == ms and ms >= 0.0:
            self._det_dq.append(ms)
            sv = sorted(self._det_dq)
            self.det_ms = sv[len(sv) // 2]

    def note_det(self, fr):
        self.n_det += 1
        self.n_ring += int(bool(fr.ring.ok))
        self.n_dot += int(any(d.ok for d in fr.dots))

    def fps(self):
        return 1.0 / (sum(self.dt) / len(self.dt)) if self.dt else 0.0

    def read_hz(self):
        if self.slot is None:
            return None
        return self.slot.n_recv / max(time.time() - self.t0, 1e-3)

    def skip_n(self):
        return self.slot.n_skip if self.slot is not None else self.dropped

    def loss_pct(self):
        tot = self.n + self.gaps
        return (100.0 * self.gaps / tot) if tot else 0.0

    def link_color(self):
        if self.has_rtt and self.rtt_p50 == self.rtt_p50:
            v, st = self.rtt_p50, self.rtt_stall
        else:
            v = 0.0 if self.lat_ms != self.lat_ms else self.lat_ms
            st = self.stalls
        if v < 250.0 and st == 0:
            return (80, 220, 80)
        if v < 1200.0:
            return (0, 200, 255)
        return (0, 0, 255)

    def _det_counts(self):
        return f"ring={self.n_ring}/{self.n_det} dot={self.n_dot}/{self.n_det}"

    def head_line(self):
        """HUD L1 (英文: OpenCV 画不了中文)"""
        if self.has_rtt:
            _lat = f"rtt={self.rtt_p50:.0f}ms"
            if self.det_ms == self.det_ms:
                _lat += f"(up={max(0.0, self.rtt_p50 - self.det_ms):.0f})"
            _stall = f"stall={self.rtt_stall}"
        else:
            lat = "--" if self.lat_ms != self.lat_ms else f"{self.lat_ms:.0f}"
            _lat = f"lat={lat}ms(rel)"
            _stall = f"stall={self.stalls}"
        _hz = self.read_hz()
        _arr = f"arrive={_hz:.1f}  " if _hz is not None else ""
        return (f"fps={self.fps():.1f}  {_arr}loss={self.loss_pct():.1f}%  "
                f"{_lat}  {_stall}  "
                f"bw={self.bytes / max(time.time() - self.t0, 1e-3) / 1024:.0f}KB/s  "
                f"{self._det_counts()}")

    def line(self):
        """控制台那行 (同 gcs_aruco_det2)"""
        _r = (f"往返={self.rtt_p50:.0f}ms(P95 {self.rtt_p95:.0f} max {self.rtt_max:.0f}) "
              if (self.has_rtt and self.rtt_p50 == self.rtt_p50) else "")
        _up = (f"上行≈{max(0.0, self.rtt_p50 - self.det_ms):.0f} "
               if (self.has_rtt and self.det_ms == self.det_ms) else "")
        _d = f"检测={self.det_ms:.0f}ms " if self.det_ms == self.det_ms else ""
        _st = self.rtt_stall if self.has_rtt else self.stalls
        return (f"n={self.n} fps={self.fps():.1f} 丢包={self.loss_pct():.1f}% "
                f"skip={self.skip_n()} drop={self.dropped} gaps={self.gaps} "
                f"{_r}{_up}{_d}Δ最快={self.lat_ms:.0f}ms(max {self.lat_max:.0f}) 卡={_st} "
                f"bw={self.bytes / max(time.time() - self.t0, 1e-3) / 1024:.0f}KB/s "
                f"环={self.n_ring}/{self.n_det} 红点={self.n_dot}/{self.n_det}")


def quat_to_rpy(q):
    """四元数 → (roll, pitch, yaw)，单位：度（HUD 显示用）"""
    r, p, y = geom.rpy_from_quat(*q, eps=1e-9)
    return math.degrees(r), math.degrees(p), math.degrees(y)


# =============================================================================
# 本帧检测的几个显示量
# =============================================================================
def ring_major_px(fr):
    """环 (中心线椭圆) 长轴, 工作分辨率像素。"""
    if fr is None:
        return None
    gL = fr.dbg.get("gL") if fr.dbg else None
    if gL is not None and gL.get("ok") and gL.get("center_major"):
        return float(gL["center_major"])
    return None


def dot_px(fr):
    d = fr.dot() if fr is not None else None
    return float(d.diam_px) if d is not None and d.diam_px else None


def dot_dist(fr):
    """红点到相机的距离 (只有 stereo / ray_plane 有; mono 只有方向)。"""
    if fr is None:
        return None
    for d in fr.dots:
        if d.ok and d.src != "mono" and d.p is not None:
            return float(np.linalg.norm(d.p))
    return None


def no_target_reason(mode, fr, fst):
    if mode == 0:
        return "mode 0"
    parts = []
    if fr is not None:
        why = fr.diag.get("ring_why")
        if why:
            parts.append(str(why))
        elif fr.diag.get("search_skip"):
            parts.append("search_skip")
        elif fr.ring_seen and not fr.ring.ok:
            parts.append("ring seen, not located (lr_ok=%s corners=%d)" % (fr.diag.get("lr_ok"), fr.ring.n_corners))
    if not fst:
        parts.append("no fused result from drone yet")
    else:
        if fst.get("pose") not in (None, "ok"):
            parts.append("pose " + str(fst.get("pose")))
        parts.append("fused " + str(fst.get("stage")))
    return "  ".join(parts) or "--"


# =============================================================================
# HUD (照 gcs_aruco_det2.draw_hud 的版面; 画在 1920×1080 画布上)
# =============================================================================
def draw_hud(show, mode, probe, st, fc, ph, fr, fst, pose, found, t_cap, src_wh, rows=None, probe_d=None):
    """原地修改 show。★纯显示，异常自己吃掉★
    rows = 检测器 hud_rows() 给的两行 [(文字, 颜色)] (本帧检测 / 飞机融合), 替换环的 L4 / L7b; probe_d = 探针大字距离。"""
    try:
        W = show.shape[1]
        ov = show.copy()
        cv2.rectangle(ov, (0, 0), (W, HUD_H), (0, 0, 0), -1)
        show[:] = cv2.addWeighted(ov, 0.55, show, 0.45, 0)
        f_ = cv2.FONT_HERSHEY_SIMPLEX
        c = (80, 220, 80) if found else (160, 160, 160)
        GRAY = (160, 160, 160)

        # ---- L1 链路统计 ----
        cv2.putText(show, f"[{MODE_NAMES[mode]}{'+PROBE' if probe else ''}] {st.head_line()}",
                    (8, 22), f_, 0.54, st.link_color() if found else GRAY, 1)

        # ---- L2 飞控模式 ----
        if fc:
            _m = fc.get("mode", "?")
            _cn, _mc = _MODE_INFO.get(_m, (_m, (230, 230, 230)))
            _t = f"FC {_m}   {'ARMED' if fc.get('armed') else 'DISARMED'}   {_cn}"
            if not fc.get("connected", False):
                _t += "   [FC NOT CONNECTED]"
                _mc = (0, 0, 255)
            cv2.putText(show, _t, (8, 46), f_, 0.58, _mc, 1)
        else:
            cv2.putText(show, "FC ----   (no /mavros/state)", (8, 46), f_, 0.54, GRAY, 1)

        # ---- L3 飞机阶段 (+ track 子模式) ----
        if ph:
            _p = ph.get("phase", "-")
            _sub = ph.get("sub_mode") or ""
            _pcn, _pc = _PHASE_INFO.get(_p, (_p, (230, 230, 230)))
            if _p == "track" and _sub:
                if _sub == "TRACK":
                    _pcn, _pc = "** VISION TRACKING **", (80, 255, 80)
                else:
                    _pcn, _pc = "manual segment - press T to hand over to vision", (0, 165, 255)
            _tail = f".{_sub}" if _sub else ""
            _pt = f"PHASE {_p}{_tail}   {_pcn}   [{'AIR' if ph.get('air') else 'GROUND'}]"
            if ph.get("abort"):
                _pt += "   [ABORT]"
                _pc = (0, 0, 255)
            elif ph.get("land"):
                _pt += "   [LAND requested]"
                _pc = (0, 120, 255)
            cv2.putText(show, _pt, (8, 70), f_, 0.60, _pc, 1)
        else:
            cv2.putText(show, "PHASE ----   (no phase topic from drone)", (8, 70), f_, 0.54, GRAY, 1)

        # ---- L4 本帧检测 (校正左目相机系: x右 y下 z前) ----
        r = fr.ring if fr is not None else None
        if rows:
            cv2.putText(show, rows[0][0], (8, 94), f_, 0.54, rows[0][1], 1)
        elif r is not None and r.ok and r.c is not None:
            s = (f"ring cam({r.c[0]:+.2f},{r.c[1]:+.2f},{r.c[2]:+.2f})  "
                 f"cam dist={np.linalg.norm(r.c):.2f}m  src={r.src}")
            if r.src == "stereo" and r.plane_rms == r.plane_rms:
                s += f"  rms={r.plane_rms * 1000:.0f}mm"
            s += f"  corners={r.n_corners}"
            if r.n is not None:
                s += f"  n:{r.n_src or '?'} tilt={ring_tilt_deg(r.c, r.n):.0f}deg"
            dd = dot_dist(fr)
            if dd is not None:
                s += f"   dot {fr.dot().src} {dd:.2f}m"
            elif fr.dot() is not None:
                s += "   dot mono"
            cv2.putText(show, s, (8, 94), f_, 0.54, c, 1)
        elif fr is not None and fr.dot() is not None:
            d = fr.dot()
            dd = dot_dist(fr)
            s = (f"ring cam(--,--,--)   dot {d.src} cam({d.p[0]:+.2f},{d.p[1]:+.2f},{d.p[2]:+.2f})  "
                 f"cam dist={dd:.2f}m" if dd is not None else
                 f"ring cam(--,--,--)   dot mono pix=({d.pix[0]:.0f},{d.pix[1]:.0f})")
            cv2.putText(show, s, (8, 94), f_, 0.54, c, 1)
        else:
            cv2.putText(show, "ring cam(--,--,--)  cam dist=--m  rms=--  corners=--", (8, 94), f_, 0.54, GRAY, 1)

        # ---- L5 目标 (飞机回传的融合结果, 世界系) ----
        if fst and fst.get("target") is not None:
            tg = fst["target"]
            sd = fst.get("sd_cm") or [0, 0, 0]
            lag = (t_cap - float(fst.get("t") or t_cap)) * 1000.0
            pubd = bool(fst.get("published"))
            s = (f"target world({tg[0]:+.2f},{tg[1]:+.2f},{tg[2]:+.2f})  "
                 f"sd=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f})cm  src={fst.get('src')}  "
                 f"{'PUB' if pubd else 'not published'}{' [SIM]' if fst.get('sim') else ''}  "
                 f"lag {lag:.0f}ms")
            cv2.putText(show, s, (8, 118), f_, 0.54, (80, 220, 80) if pubd else (0, 200, 255), 1)
        else:
            cv2.putText(show, f"no target: {no_target_reason(mode, fr, fst)}", (8, 118), f_, 0.54, GRAY, 1)

        # ---- L6 飞机位置 ----
        if pose:
            p = pose["p"]
            d_org = math.hypot(p[0], p[1])
            cv2.putText(show, f"drone({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})  "
                              f"from origin={d_org:.2f}m  alt={p[2]:.2f}m",
                        (8, 142), f_, 0.54, (200, 200, 200), 1)

        # ---- L7 姿态 (按倾角变色) ----
        if pose:
            _rr, _pp, _yy = quat_to_rpy(pose["q"])
            _tilt = max(abs(_rr), abs(_pp))
            _tc = ((0, 0, 255) if _tilt > 25.0 else (0, 200, 255) if _tilt > 15.0 else (200, 200, 200))
            cv2.putText(show, f"att R={_rr:+.1f}  P={_pp:+.1f}  Y={_yy:+.1f} (deg)",
                        (8, 166), f_, 0.54, _tc, 1)
        elif probe:
            cv2.putText(show, "probe mode: no VIO needed, only checks detection / range",
                        (8, 142), f_, 0.54, (0, 220, 255), 1)

        # ---- L7b 源图 + 像素尺寸 + 阶段 ----
        _src = f"src {src_wh[0]}x{src_wh[1]} (eye {src_wh[0] // 2}x{src_wh[1]})   " if src_wh else ""
        if rows and len(rows) > 1:
            cv2.putText(show, rows[1][0], (8, 190), f_, 0.54, rows[1][1], 1)
        else:
            _rp, _dp = ring_major_px(fr), dot_px(fr)
            _stage = f"   stage={fr.stage}" if fr is not None else ""
            cv2.putText(show, f"{_src}ring px={'--' if _rp is None else f'{_rp:.0f}'}   "
                              f"dot px={'--' if _dp is None else f'{_dp:.0f}'}{_stage}",
                        (8, 190), f_, 0.54, (200, 200, 120), 1)

        # ---- L8 两个核心数字 ----
        if probe:
            _rd = float(np.linalg.norm(r.c)) if (r is not None and r.ok and r.c is not None) else None
            _dd = probe_d if rows else dot_dist(fr)
            _s1 = f"ring dist = {_rd:.2f} m" if _rd is not None else "ring dist = -- m"
            _s2 = f"dot dist = {_dd:.2f} m" if _dd is not None else "dot dist = -- m"
            _c1 = (0, 255, 255) if _rd is not None else GRAY
            _c2 = (0, 255, 255) if _dd is not None else GRAY
        else:
            tg = fst.get("target") if fst else None
            if pose and tg is not None:
                p = pose["p"]
                d_xy = math.hypot(tg[0] - p[0], tg[1] - p[1])
                err_z = tg[2] - p[2]
                _s1, _c1 = f"horiz d = {d_xy:.2f} m", (0, 255, 255)
                _s2 = f"vert err_z = {err_z:+.2f} m"
                _c2 = (0, 255, 0) if abs(err_z) < HUD_Z_DEADBAND else (0, 200, 255)
            else:
                _s1, _c1 = "horiz d = -- m", GRAY
                _s2, _c2 = "vert err_z = -- m", GRAY
        cv2.putText(show, _s1, (8, 208), f_, 0.74, _c1, 2)
        (w1, _), _ = cv2.getTextSize(_s1, f_, 0.74, 2)
        cv2.putText(show, _s2, (8 + w1 + 44, 208), f_, 0.74, _c2, 2)
    except Exception as _e:
        print(f"[gcs_gate] HUD 绘制异常(不影响检测): {type(_e).__name__}: {_e}")


def to_canvas(img):
    """检测画面 (960×540, 或 --show-mask 时 1920×540) → 高 1080 的显示画布。"""
    h = img.shape[0]
    if h == OUT_H:
        return img.copy()
    s = OUT_H / float(h)
    return cv2.resize(img, (int(round(img.shape[1] * s)), OUT_H), interpolation=cv2.INTER_LINEAR)


# =============================================================================
# 控制台 / 网页
# =============================================================================
def fused_lines(fst, t_cap):
    """飞机回传的融合结果 (上一帧的, 有一个往返的滞后) → 控制台文字。"""
    if not fst:
        return ["FUSED (飞机) --  还没收到融合结果"]
    lag = (t_cap - float(fst.get("t") or t_cap)) * 1000.0
    s = "FUSED [%s] pose=%s" % (str(fst.get("stage", "?")).upper(), fst.get("pose"))
    if fst.get("target") is not None:
        sd = fst.get("sd_cm") or [0, 0, 0]
        s += " dist=%.2fm sd=(%.1f,%.1f,%.1f)cm src=%s %s" % (
            float(fst.get("dist") or 0), sd[0], sd[1], sd[2], fst.get("src"),
            "PUB" if fst.get("published") else "")
        if fst.get("sim"):
            s += " [SIM]"
    return [s + "  (滞后 %.0fms)" % lag]


def preview_meta(args, sim, st, fr, fst, head, jpg):
    """网页 X-Det: kind=gate (字段同改之前, 网页那边不用动) + 老版的 probe/reason/px。"""
    m = {
        "kind": "gate", "mode": "OFF" if args.mode == 0 else ("SIM" if sim else "REAL"),
        "probe": bool(args.probe),
        "fps": st.fps(), "loss": st.loss_pct(),
        "det_ms": st.det_ms if st.det_ms == st.det_ms else None,
        "rtt": st.rtt_p50 if st.has_rtt else None,
        "stall": st.rtt_stall if st.has_rtt else st.stalls,
        "up_w": head.get("w"), "up_h": head.get("h"), "up_kb": len(jpg) / 1024.0,
    }
    if fr is not None:
        r = fr.ring
        m.update({
            "found": bool(r.ok or any(d.ok for d in fr.dots)),
            "g_stage": fr.stage, "ring": r.src if r.ok else None, "corners": r.n_corners,
            "dots": [d.src for d in fr.dots if d.ok],
            "z_ring": fr.diag.get("z_ring"), "plane_rms": fr.diag.get("plane_rms"),
            "n_src": r.n_src if r.ok else None, "tilt": fr.diag.get("tilt") if r.ok else None,
            "px": ring_major_px(fr),
        })
    else:
        m["found"] = False
    m["reason"] = "" if m["found"] else no_target_reason(args.mode, fr, fst)
    if fst:
        m.update(stage=fst.get("stage"), dist=fst.get("dist"), src=fst.get("src"),
                 sd_cm=fst.get("sd_cm"), published=fst.get("published"), pose=fst.get("pose"),
                 sim=fst.get("sim"))
    else:
        m["stage"] = fr.stage if fr is not None else None
    return m


def _lan_ips(drone):
    """本机的 IPv4 (给"别的电脑怎么填地址"的提示用)。拿不到就返回空。"""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    try:   # 不发包: UDP connect 只是让系统选出口网卡
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((drone or "192.168.2.20", 9))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def _dpi_aware():
    """Windows 显示缩放 125%/150% 时防止 OpenCV 窗口被系统整窗拉伸变糊 (同 gcs_aruco_det2)。"""
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="地面站检测 (飞机 vision_bridge _mode:=gate_gcs): 格子环+红点 (--target ring) / 红点 (--target reddot)")
    # ---- 和 gcs_aruco_det2 相同的参数 ----
    ap.add_argument("--drone", help="飞机 IP (--replay 时不用)")
    ap.add_argument("--mode", type=int, default=1, choices=[0, 1, 2],
                    help="0=不检测(只看画面) 1=检测,sim(默认) 2=检测,★驱动飞机")
    ap.add_argument("--pitch", type=float, default=None, help="gate 版不起作用: 外参在飞机上 (_cam_pitch_deg)")
    ap.add_argument("--show-color", action="store_true", help="gate 版不起作用: 画面固定是校正后左目彩色图")
    ap.add_argument("--roll", type=float, default=None, help="gate 版不起作用: 外参在飞机上 (_cam_roll_deg)")
    ap.add_argument("--with-sub", action="store_true", help="gate 版不起作用: 只对 ArUco 板有效")
    ap.add_argument("--probe", action="store_true",
                    help="探针模式: 强制 sim=true (不驱动飞机), 大字显示 环距/红点距")
    ap.add_argument("--headless", action="store_true", help="不开本地窗口")
    ap.add_argument("--no-preview", action="store_true", help="不开网页预览服务")
    ap.add_argument("--preview-host", default="127.0.0.1",
                    help="预览监听地址；别的电脑也要看时用 0.0.0.0")
    ap.add_argument("--preview-port", type=int, default=8091)
    ap.add_argument("--preview-gray", action="store_true", help="gate 版不起作用: 画面固定是校正后左目彩色图")
    ap.add_argument("--preview-width", type=int, default=960, help="网页预览缩放宽度；跨 WiFi 看用 480")
    # ---- gate 版自己的 ----
    ap.add_argument("--calib", default=os.path.join(HERE, "stereo_calib.yaml"))
    ap.add_argument("--params", default=os.path.join(HERE, "gate_params.json"))
    ap.add_argument("--target", default="ring", choices=registry.kinds(registry.DETECTORS),
                    help="目标类型 (vb09/perception/registry.py), 默认 ring。飞机桥的 _target:= 要一样")
    ap.add_argument("--scale", type=float, default=0.5, help="检测工作分辨率 (相对 1920×1080/目), 默认 0.5")
    ap.add_argument("--record", default=None, help="把收到的上行包存到这个目录 (每 --record-every 帧一张)")
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--replay", default=None, help="不连飞机, 回放 --record 存的目录")
    ap.add_argument("--replay-hz", type=float, default=10.0)
    ap.add_argument("--show-mask", action="store_true",
                    help="画面右边拼上饱和度掩码 (白 = 检测器认为是彩色的像素), 调 s_min/v_min 用")
    ap.add_argument("--advice", action="store_true",
                    help="画面底部叠加「第二意见」信息条 (需 advice_overlay.py + 建议 JSON); 默认关")
    ap.add_argument("--advice-file", default=None,
                    help="建议 JSON 路径 (指定时自动启用 --advice; 默认 $TMPDIR/cpmv_advice.json; 也可用 CPMV_ADVICE)")
    ap.add_argument("--real", action="store_true", help=argparse.SUPPRESS)     # 旧写法 = --mode 2
    args = ap.parse_args()
    if not args.drone and not args.replay:
        ap.error("要么 --drone <飞机IP>, 要么 --replay <目录>")
    if zmq is None and not args.replay:
        sys.exit("pip install pyzmq")

    # ---- 建议叠加: 默认关, 只有显式 --advice / --advice-file 才按需加载启用 ----
    draw_advice_fn = None
    if args.advice or args.advice_file:
        draw_advice_fn, adv_dir = _load_advice_overlay()
        if draw_advice_fn is None:
            print("[gcs_gate] --advice 指定了, 但 advice_overlay.py 加载失败 — 忽略, 画面和以前一样")
        else:
            print("[gcs_gate] 建议叠加已启用 (%s): %s"
                  % (args.advice_file or "默认路径", adv_dir or "-"))

    # ---- 参数提示 (老参数在 gate 版不起作用的, 只提示不报错) ----
    if args.real:
        args.mode = 2
        print("[gcs_gate] --real 已改名为 --mode 2 (按 --mode 2 处理)")
    if args.pitch is not None or args.roll is not None:
        print("[gcs_gate] ! --pitch/--roll 在 gate 版不起作用 (外参在飞机上), 起桥时给: "
              "bash run_vision.sh <地面站IP> _mode:=gate_gcs _cam_pitch_deg:=%s _cam_roll_deg:=%s"
              % (args.pitch if args.pitch is not None else 1.3, args.roll if args.roll is not None else 0))
    if args.with_sub:
        print("[gcs_gate] ! --with-sub 只对 ArUco 板有效, 忽略")
    if args.show_color or args.preview_gray:
        print("[gcs_gate] ! 画面固定是校正后左目彩色图 (= 检测器看到的), --show-color/--preview-gray 忽略")
    sim = (args.mode != 2) or args.probe          # 同 gcs_aruco_det2: probe 永远 sim

    rig = StereoRig.from_file(args.calib, (1920, 1080), args.scale)
    cfg, src = registry.load_config(args.target, args.params)
    src = src or "默认参数 (没找到 %s)" % args.params
    det = registry.load(registry.DETECTORS, args.target)(rig, cfg)
    # 2026-11-05: 检测器自带地面显示钩子 (红点) → HUD / 叠加 / 控制台 / 掩膜用钩子; 没有 → 下面的环代码
    hooks = all(hasattr(det, k) for k in ("hud_rows", "ground_draw", "console", "mask", "found"))
    if args.target != "ring" and not hooks:
        sys.exit("[gcs_gate] --target %s: 检测器没有地面显示钩子 (hud_rows / ground_draw / console / mask / found), "
                 "本程序的 HUD 只会画环" % args.target)
    codec = wire.codec(args.target)
    rig_msg = rig_to_msg(rig, cfg, args.target)
    print(f"[gcs_gate] 标定 {args.calib}: 工作分辨率 {rig.size[0]}x{rig.size[1]}/目 f={rig.f:.1f} "
          f"B={rig.B * 1000:.1f}mm (T 单位 {rig.T_unit})  参数: {src}")
    print(f"[gcs_gate] 目标 {args.target}  检测库 {rig_msg['lib']}  GateConfig {rig_msg['cfg']} "
          f"(飞机上的桥核对这两个; 不一致 → 飞机强制 SIM, 网页报 \"版本不符\")")
    print(f"[gcs_gate] 模式 {args.mode} = {MODE_NAMES[args.mode]}   回包 sim={sim}")
    if args.mode == 2 and not args.probe:
        print("[gcs_gate] ⚠ mode 2 会发 sim=false！仅飞控确认后使用！")
    if args.probe:
        print("[gcs_gate] ★ PROBE 探针模式：地面检测本来不需要位姿；强制 sim=true，不会驱动飞机")
    dec_flag = {0.5: cv2.IMREAD_REDUCED_COLOR_2, 0.25: cv2.IMREAD_REDUCED_COLOR_4}.get(args.scale)

    # ---- 网络 ----
    slot, stop = FrameSlot(), threading.Event()
    pub = None
    if args.replay:
        threading.Thread(target=replay_loop, args=(args.replay, slot, stop, args.replay_hz), daemon=True).start()
    else:
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVHWM, 4)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.RCVBUF, 65536)       # 内核接收缓冲封顶 (同 gcs_aruco_det2)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(f"tcp://{args.drone}:{UP_PORT}")
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 4)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f"tcp://*:{DOWN_PORT}")
        threading.Thread(target=reader_loop, args=(sub, slot, stop), daemon=True).start()
        print(f"[gcs_gate] 上行 SUB tcp://{args.drone}:{UP_PORT}   下行 PUB tcp://*:{DOWN_PORT}")

    win = "gcs_gate"
    if not args.headless:
        _dpi_aware()                       # ★必须在建窗口之前
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # ---- 网页预览 (同 gcs_aruco_det2: 端口被占 / 文件不在 → 只是网页看不到, 检测照常) ----
    pv = None
    pv_full = False
    preview_server = None
    if not args.no_preview:
        try:
            from vb09.common import preview_server
            import inspect
            _ps = inspect.signature(preview_server.publish).parameters
            pv_full = "meta" in _ps and "raw" in _ps
            print(f"[gcs_gate] preview_server: {os.path.abspath(preview_server.__file__)}")
            if not pv_full:
                print("[gcs_gate] ★ 这是**旧版** preview_server.py (publish 不认 meta/raw) —— 网页只有画面, "
                      "没有检测数字、没有「原始」画质。换成新版 preview_server.py")
            pv = preview_server.start(port=args.preview_port, host=args.preview_host,
                                      max_width=args.preview_width)
        except ImportError:
            print("[gcs_gate] 没找到 preview_server.py —— 网页看不到识别画面, 检测不受影响。把它放到 %s" % HERE)
    if pv is not None:
        print("[gcs_gate] 网页「识别画面 → 地址」:")
        print(f"        浏览器在**这台**电脑:  留空 (= http://127.0.0.1:{args.preview_port}/snapshot.jpg)")
        if args.preview_host in ("0.0.0.0", ""):
            for ip in _lan_ips(args.drone) or ["<本机IP>"]:
                print(f"        浏览器在**另一台**电脑: http://{ip}:{args.preview_port}/snapshot.jpg")
        else:
            print(f"        浏览器在另一台电脑: 现在**看不了** (只绑了 {args.preview_host}) —— "
                  f"加 --preview-host 0.0.0.0 重启本程序")
        print(f"        自测: 本机浏览器直接打开 http://127.0.0.1:{args.preview_port}/  应该能看到画面")

    def pv_publish(img, meta, jpg):
        if pv is None:
            return
        try:
            if pv_full:
                preview_server.publish(img, meta=meta, raw=jpg)
            else:
                preview_server.publish(img)
        except Exception as _e:      # 同一个错只打一次
            if str(_e) != getattr(main, "_pv_err", None):
                main._pv_err = str(_e)
                print(f"[gcs_gate] ⚠ 预览 publish 异常(忽略, 同样的错不再重复打印): {_e}")

    if args.record:
        os.makedirs(args.record, exist_ok=True)

    st = Stats()
    st.slot = slot
    warned = set()
    t_print = t_start = t_read_report = time.time()
    read_last = (t_read_report, 0, 0)
    n_rec = 0
    try:
        while not stop.is_set():
            raw = slot.get(timeout=0.5)
            if raw is None:
                if "noframe" not in warned and time.time() - t_start > 5.0 and st.n == 0:
                    print("[gcs_gate] 5s 没收到任何帧 —— 飞机上 bash run_vision.sh <本机IP> _mode:=gate_gcs 起了吗")
                    warned.add("noframe")
                continue

            # 每 5 秒汇报一次"到货率 vs 被跳过" (同 gcs_aruco_det2)
            if time.time() - t_read_report > 5.0:
                _t = time.time()
                _pt, _pn, _ps = read_last
                if _t - _pt >= 1.0:
                    print("[gcs_gate] 收图: 到货 %.1f Hz, 其中被跳过 %.1f Hz（检测比到货慢多少）"
                          % ((slot.n_recv - _pn) / (_t - _pt), (slot.n_skip - _ps) / (_t - _pt)))
                read_last = (_t, slot.n_recv, slot.n_skip)
                t_read_report = _t

            try:
                head = unpack(raw)
                jpg = head["jpg"]
            except Exception as e:
                print(f"[gcs_gate] 解包失败: {e}")
                continue
            st.tick(int(head.get("seq") or 0), len(raw), head.get("t_cap"), head=head)
            if args.record and st.n % max(1, args.record_every) == 0:
                with open(os.path.join(args.record, "%08d.msgpack" % st.n), "wb") as fh:
                    fh.write(raw)
                n_rec += 1

            pose = head.get("pose")
            fc = head.get("fc") or {}
            ph = {}
            if head.get("ph"):
                try:
                    ph = json.loads(head["ph"])
                except Exception:
                    ph = {}
            fst = head.get("gate")                       # 飞机最新的融合结果 (显示用)
            w, h = int(head.get("w") or 0), int(head.get("h") or 0)
            need_img = (not args.headless) or pv is not None

            if not head.get("sbs"):
                if "notsbs" not in warned:
                    print("[gcs_gate] ★ 收到的不是并排双目帧 (src=%s %dx%d) —— 飞机桥要用 _mode:=gate_gcs。"
                          "现在只显示画面, 不回观测" % (head.get("src"), w, h))
                    warned.add("notsbs")
                if need_img:
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        pv_publish(img, {"kind": "gate", "found": False,
                                         "reason": "飞机桥不是 gate_gcs 模式"}, jpg)
                        if not args.headless:
                            cv2.imshow(win, img)
                            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                break
                continue

            # 并排帧: 每目宽 = w/2。整帧原样 3840 宽时用 DCT 域降采样解码 (快)
            flag = dec_flag if (dec_flag is not None and w >= 2 * rig.size_full[0]) else cv2.IMREAD_COLOR
            img = None
            fr = None
            t_cap = float(head.get("t_cap") or time.time())
            if args.mode != 0 or need_img:
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), flag)
                if img is None:
                    continue
                eye_ar = (img.shape[1] / 2.0) / img.shape[0]
                if abs(eye_ar - 16 / 9.0) > 0.05 and "ar" not in warned:
                    print("[gcs_gate] ★ 每目宽高比 %.3f ≠ 16:9 —— 不是这颗相机的并排帧, 结果不可信" % eye_ar)
                    warned.add("ar")
            if args.mode != 0:
                L, R = rig.rectify(*rig.split(img))
                fr = det.process(L, R, t_cap, None)          # 不给位姿: 只出相机系观测
                st.note_det_ms(fr.t_ms.get("total", 0.0))
                if hooks:
                    st.n_det += 1
                    st.n_dot += int(det.found(fr))
                else:
                    st.note_det(fr)
            elif img is not None:
                L = rig.rectify(*rig.split(img))[0]           # mode 0: 只给人看
            if hooks:
                found = bool(fr is not None and det.found(fr))
            else:
                found = bool(fr is not None and (fr.ring.ok or any(d.ok for d in fr.dots)))

            if pub is not None:
                msg = {"seq": head.get("seq", 0), "t_cap": head.get("t_cap"), "t_done": time.time(),
                       "found": found, "target": None,      # gate 结果不走旧的世界系 target 字段
                       "sim": sim, "mode": args.mode, "model": "gate/" + MODE_NAMES[args.mode]}
                if fr is not None:                           # mode 0: 不带 gate → 桥只统计往返
                    msg["gate"], msg["rig"] = codec.frame_to_msg(fr), rig_msg
                try:
                    pub.send(pack(msg), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass

            if need_img:
                # 检测标记画在 960×540 工作图上, 不写字 (同 gcs_aruco_det2: 网页图不画 HUD)
                if hooks:
                    vis = det.ground_draw(L, fr, fst) if fr is not None else L.copy()
                else:
                    vis = det.draw(L, fr, None, text=False) if fr is not None else L.copy()
                # 第二意见叠加: 在 --show-mask 拼接**之前**画
                #   （画在拼接之后的话, 归一化坐标会按拼接后的双倍宽度换算, 框会被横向拉长 2 倍）
                if draw_advice_fn is not None:
                    try:
                        vis = draw_advice_fn(vis, args.advice_file)
                    except Exception as _e:
                        if str(_e) != globals().get("_adv_err"):
                            globals()["_adv_err"] = str(_e)
                            print("[gcs_gate] ! 建议叠加异常(忽略): %s" % _e)
                if args.show_mask and fr is not None:
                    if hooks:
                        vis = np.hstack([vis, det.mask(L)])
                    else:
                        m_ = segment(apply_wb(L, cfg), cfg)["sat"]
                        vis = np.hstack([vis, cv2.cvtColor(m_, cv2.COLOR_GRAY2BGR)])
                if pv is not None:
                    meta = preview_meta(args, sim, st, fr, fst, head, jpg)
                    if hooks and fr is not None:
                        meta.update(found=found, n_cand=len(getattr(fr, "cands", []) or []),
                                    reason="" if found else no_target_reason(args.mode, fr, fst))
                    if hooks:
                        # 网页点选 (2026-11-08): 这张图 (校正左目, 工作分辨率) 的内参 + 拍摄时刻 → 网页算视线
                        meta["pick"] = {"f": float(rig.f), "cx": float(rig.cx), "cy": float(rig.cy),
                                        "w": int(L.shape[1]), "h": int(L.shape[0]), "img_w": int(vis.shape[1]),
                                        "t_cap": t_cap}
                        if fst:
                            meta.update(dot_sel=fst.get("sel"), dot_cands=[
                                {"id": c.get("id"), "uv": c.get("uv"), "score": c.get("score")}
                                for c in (fst.get("cands") or [])])
                    pv_publish(vis, meta, jpg)
                if not args.headless:
                    show = to_canvas(vis)
                    rows = det.hud_rows(fr, fst) if hooks else None
                    pd = det.probe_dist(fr) if (hooks and hasattr(det, "probe_dist")) else None
                    draw_hud(show, args.mode, args.probe, st, fc, ph, fr, fst, pose, found, t_cap, (w, h),
                             rows=rows, probe_d=pd)
                    cv2.imshow(win, show)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

            if time.time() - t_print > 2.0:
                t_print = time.time()
                up = f" 上行={w}x{h} {len(jpg) / 1024:.0f}KB/张"
                if fr is None:
                    print(f"[gcs_gate] {st.line()}  found=False (mode 0){up}")
                elif hooks:
                    print(f"[gcs_gate] {st.line()}  found={found} 阶段(飞机)={(fst or {}).get('stage', '--')} "
                          f"{det.console(fr, fst)}{up}"
                          + (f" | 已存 {n_rec} 帧" if args.record else ""))
                    print("           耗时 %s" % {k: round(v) for k, v in fr.t_ms.items()})
                    if fr.diag:
                        print("           diag " + "  ".join("%s=%s" % (k, v) for k, v in fr.diag.items()))
                else:
                    r_, d_ = fr.ring, fr.diag
                    dd = f" d={np.linalg.norm(r_.c):.2f}m" if (r_.ok and r_.c is not None) else ""
                    rp = ring_major_px(fr)
                    px = f" 环px={rp:.0f}" if rp else ""
                    rms = (f" rms={r_.plane_rms * 1000:.0f}mm"
                           if (r_.ok and r_.src == "stereo" and r_.plane_rms == r_.plane_rms) else "")
                    dot = "".join(f" 红点={x.src}" for x in fr.dots if x.ok)
                    rsn = f" ({d_.get('ring_why')})" if (not found and d_.get("ring_why")) else ""
                    print(f"[gcs_gate] {st.line()}  found={found} 阶段={fr.stage}{dd}{px}{rms}{dot}{rsn}{up}"
                          + (f" | 已存 {n_rec} 帧" if args.record else ""))
                    print("           耗时 %s" % {k: round(v) for k, v in fr.t_ms.items()})
                    if d_:
                        print("           diag " + "  ".join("%s=%s" % (k, v) for k, v in d_.items()))
                    if not fr.ring_seen and d_.get("ring_why"):
                        print("           环没找到: %s  sat_frac=%s sq_est=%s close_k=%s  最好的被拒假设: onb=%s(≥%.2f) "
                              "cov=%s(≥%.2f) hole=%s(≤%.2f) w=%s(0.5~1.6) 长轴=%spx"
                              % (d_.get("ring_why"), d_.get("sat_frac"), d_.get("sq_est"), d_.get("close_k"),
                                 d_.get("rej_onb"), cfg.min_onband, d_.get("rej_cov"), cfg.min_coverage,
                                 d_.get("rej_hole"), cfg.max_hole, d_.get("rej_w"), d_.get("rej_px")))
                    if args.probe:
                        rc = r_.c if (r_.ok and r_.c is not None) else None
                        ddot = dot_dist(fr)
                        print("[probe] 校正左目系 (x右, y下, z前)  环心=%s  环距=%s  红点距=%s"
                              % ("(%+.3f, %+.3f, %+.3f) m" % tuple(rc) if rc is not None else "--",
                                 "%.3f m" % np.linalg.norm(rc) if rc is not None else "--",
                                 "%.3f m" % ddot if ddot is not None else "--"))
                for ln in fused_lines(fst, t_cap):
                    print("           " + ln)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if not args.headless:
            cv2.destroyAllWindows()
        if pv is not None:
            try:
                preview_server.stop()
            except Exception:
                pass
        print(f"\n[gcs_gate] 收工。{st.line()}")


if __name__ == "__main__":
    main()
