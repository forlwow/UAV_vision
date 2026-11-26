# -*- coding: utf-8 -*-
"""
vb09/flight/phases.py —— 起飞 / 下降 / AUTO.LAND / 接管 / cleanup 判定 / 解锁 / 单实例 / 紧急降落 / 位姿回调
===============================================================================
2026-10-30 从 stage12_gate.py 原样搬来 (第 4.2 步, tools/dev/split_stage12.py 生成);
共享状态改成 F.名字 (vb09/flight/ctx.py 的 FlightContext 单例), 参数是 C.名字。
"""

import contextlib
import sys
import math
import threading
import time

import rospy
from vb09.flight import config as C

from vb09.flight.ctx import F
from vb09.flight.ctx import _MAV_RESULT, check_abort, fcu_texts_since, request_abort, request_land, set_phase
from vb09.flight.devices import KeyReader, _flush_stdin
from vb09.flight.log import timing_queue, timing_queue_lock


@contextlib.contextmanager
def _safety_keys(tag):
    """
    为起飞 / 缓降 / AUTO.LAND 提供阶段级 cbreak 键盘监听 (X 停桨 / L 降落 / EOF 断连)。
    每个阶段独立进入/退出 KeyReader，保证任何退出路径都恢复终端属性，不新增模块级可变全局。
    非 tty (如离线仿真) 静默返回 None。
    """
    keys = None
    if sys.stdin.isatty():
        try:
            keys = KeyReader()
            keys.__enter__()
        except Exception as e:
            rospy.logwarn(f"[keys] {tag} 阶段安全键读取器创建失败 (X/L 不可用): {e}")
            keys = None
    try:
        yield keys
    finally:
        if keys is not None:
            keys.__exit__(None, None, None)


def _poll_safety_keys(keys, tag, allow_land=True):
    """
    轮询一次安全键。返回 (keys_next, hit):
      · X   → request_abort (立即停桨)
      · EOF → request_land (allow_land=True 时触发降落)，同时返回 keys_next=None 避免重复读已关闭的 stdin
      · L   → allow_land=True 时 request_land；已在降落流程中则仅提示
    """
    if keys is None:
        return None, False
    try:
        _, flags, eof, _ = keys.poll()
    except Exception:
        return keys, False
    if 'x' in flags:
        rospy.logerr(f"[keys] ★ {tag} 阶段收到 X — 立即停桨")
        request_abort(f"{tag} X 键")
        return keys, True
    if eof:
        if allow_land:
            rospy.logerr(f"[keys] ★ {tag} 阶段 stdin EOF (SSH 断连?) — 请求降落")
            request_land(f"{tag} stdin EOF")
            return None, True
        rospy.logwarn(f"[keys] {tag} 阶段 stdin EOF (SSH 断连?) — 已在降落流程中")
        request_land(f"{tag} stdin EOF")
        return None, False
    if 'l' in flags:
        if allow_land:
            rospy.logwarn(f"[keys] ★ {tag} 阶段收到 L — 请求降落")
            request_land(f"{tag} L 键")
            return keys, True
        rospy.loginfo(f"[keys] {tag} 阶段收到 L — 已在降落流程中 (紧急停桨请按 X)")
    return keys, False


def force_disarm_with_timeout(cmd_long, overall_timeout=2.0):
    result = {"ok": False}

    def _worker():
        for i in range(3):
            try:
                r = cmd_long(broadcast=False, command=400, confirmation=0,
                             param1=0.0, param2=21196.0,
                             param3=0.0, param4=0.0,
                             param5=0.0, param6=0.0, param7=0.0)
                rospy.logwarn(f"force_disarm attempt {i+1}: success={r.success}")
                if r.success:
                    result["ok"] = True
                    return
            except Exception as e:
                rospy.logerr(f"force_disarm attempt {i+1} exception: {e}")
            rospy.sleep(0.1)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=overall_timeout)
    if t.is_alive():
        rospy.logerr(f"force_disarm timed out after {overall_timeout}s")
    return result["ok"]


# =============================================================================
# ================================   Phases   =================================
# =============================================================================
def phase_takeoff():
    """ZHOLD 位置模式爬升。xy 目标锁在 origin。"""
    set_phase("takeoff")
    target_z = F.INITIAL_Z + C.TAKEOFF_HEIGHT
    rospy.loginfo(f"[takeoff] ZHOLD xy=origin z={target_z:.2f} "
                  f"(dz=+{C.TAKEOFF_HEIGHT}) timeout={C.TAKEOFF_TIMEOUT}s")
    F.setpoint.set_zhold_mode(F.INITIAL_X, F.INITIAL_Y, target_z, F.INITIAL_YAW, 0.0, 0.0)

    reach_frames_target = max(1, int(round(C.TAKEOFF_REACH_S * C.PHASE_LOOP_HZ)))
    t_end = rospy.Time.now() + rospy.Duration(C.TAKEOFF_TIMEOUT)
    t_log = rospy.Time.now()
    reach_count = 0
    rate = rospy.Rate(C.PHASE_LOOP_HZ)

    with _safety_keys("takeoff") as keys:
        while rospy.Time.now() < t_end:
            keys, _ = _poll_safety_keys(keys, "takeoff", allow_land=True)
            check_abort()
            if F.land_flag.is_set():
                rospy.logwarn("[takeoff] LAND 请求 — 停止爬升")
                return target_z
            zf = F.tele.z_filtered()
            if zf is not None:
                dz = zf - F.INITIAL_Z
                if not F.in_air[0] and dz > C.TAKEOFF_LIFTOFF_DZ:
                    F.in_air[0] = True
                    rospy.logwarn(f"[takeoff] LIFTOFF DETECTED at dz={dz:.2f}")

                if abs(zf - target_z) < C.TAKEOFF_REACH_TOL:
                    reach_count += 1
                    if reach_count >= reach_frames_target:
                        rospy.loginfo(f"[takeoff] REACHED z={zf:.2f} "
                                      f"(target {target_z:.2f})")
                        return target_z
                else:
                    reach_count = 0

                if (rospy.Time.now() - t_log).to_sec() >= 0.2:
                    px, py = F.tele.xy()
                    v = F.tele.velocity.twist.linear if F.tele.velocity else None
                    dx, dy = px - F.INITIAL_X, py - F.INITIAL_Y
                    if v:
                        rospy.loginfo(f"    d=({dx:+.2f},{dy:+.2f},{dz:+.2f}) "
                                      f"v=({v.x:+.2f},{v.y:+.2f},{v.z:+.2f}) "
                                      f"err_z={target_z-zf:+.2f} "
                                      f"reach_cnt={reach_count}")
                    else:
                        rospy.loginfo(f"    d=({dx:+.2f},{dy:+.2f},{dz:+.2f}) "
                                      f"reach_cnt={reach_count}")
                    t_log = rospy.Time.now()
            rate.sleep()

    zf = F.tele.z_filtered()
    dz_final = (zf - F.INITIAL_Z) if zf is not None else 0.0
    if F.in_air[0]:
        rospy.logwarn(f"[takeoff] TIMEOUT but in air at dz={dz_final:+.2f}")
        return target_z
    rospy.logerr(f"[takeoff] TIMEOUT without liftoff (dz={dz_final:+.2f})")
    raise RuntimeError("takeoff timeout without liftoff")


def phase_descend(z_from, yaw_hold, hold_x, hold_y):
    """
    ZHOLD 缓降。xy 锁在 hold_x/hold_y (不一定是 origin)。

    · 包络按进入时的实际位置动态放宽 —— 见 set_phase 的注释。
    · TRACK_Z_MIN 的钳位只在 track 阶段生效，这里不适用: 降落本来就要穿过那条线。
    """
    px, py = F.tele.xy()
    horiz_now = math.hypot(px - F.INITIAL_X, py - F.INITIAL_Y)
    horiz_lim = max(C.PHASE_ENVELOPE["descend"][0], horiz_now + C.ENVELOPE_MARGIN)
    set_phase("descend", horiz=horiz_lim)

    z_target_final = F.INITIAL_Z + C.DESCEND_Z_TARGET_DZ
    rospy.loginfo(f"[descend] ZHOLD ramp z: {z_from:.2f} → {z_target_final:.2f} "
                  f"({C.DESCEND_SPEED}m/s), xy=({hold_x:+.2f},{hold_y:+.2f}) "
                  f"包络 {horiz_lim:.2f}m")

    # 从"指令高度"和"实际高度"里取低的那个起步。LAND 路径下 z_from 可能
    # 是个很旧的值 (比如起飞阶段就触发了 LAND)，从那里往下斜坡会白等几秒。
    zf0 = F.tele.z_filtered()
    # 已贴地 / 不高于降落目标时，下面的 max(..., z_target_final) 会把设定点往上拉到
    # +DESCEND_Z_TARGET_DZ，导致电机持续爬升/堵转直到超时 → 直接转 auto_land。
    if zf0 is not None and zf0 <= z_target_final + C.DESCEND_REACH_TOL:
        rospy.logwarn(f"[descend] 实际高度 {zf0:.2f} 已不高于降落目标 {z_target_final:.2f} "
                      f"(容差 {C.DESCEND_REACH_TOL:.2f}) — 直接进 auto_land，不往上拉")
        return
    target_z = min(z_from, zf0) if zf0 is not None else z_from
    rate = rospy.Rate(20)
    last = rospy.Time.now()
    t_end = rospy.Time.now() + rospy.Duration(C.DESCEND_TIMEOUT)
    t_log = rospy.Time.now()

    with _safety_keys("descend") as keys:
        while rospy.Time.now() < t_end:
            keys, _ = _poll_safety_keys(keys, "descend", allow_land=False)
            check_abort()
            if F.takeover[0] is not None:
                rospy.logwarn("[descend] 控制权被接管 — 停止下降, 不再切 AUTO.LAND")
                return
            now = rospy.Time.now()
            dt = min((now - last).to_sec(), 0.1)
            last = now

            target_z = max(target_z - C.DESCEND_SPEED * dt, z_target_final)
            F.setpoint.set_zhold_mode(hold_x, hold_y, target_z, yaw_hold, 0.0, 0.0)

            if (now - t_log).to_sec() >= 0.3:
                zf = F.tele.z_filtered()
                dz_real = (zf - F.INITIAL_Z) if zf is not None else float('nan')
                rospy.loginfo(f"    target_dz={target_z-F.INITIAL_Z:+.2f}  "
                              f"real_dz={dz_real:+.2f}")
                t_log = now

            if target_z <= z_target_final + 1e-3:
                zf = F.tele.z_filtered()
                if zf is not None and zf <= z_target_final + C.DESCEND_REACH_TOL:
                    rospy.loginfo("[descend] 到达低高度，准备 auto_land")
                    return
            rate.sleep()

    rospy.logwarn("[descend] TIMEOUT — 强行进入 auto_land")


def auto_land_timeout():
    """
    AUTO.LAND 等待时间, 按当前高度放大 (下限 AUTO_LAND_TIMEOUT)。
    descend 超时后可能从高处切 AUTO.LAND, 写死 15s 会在 PX4 还没降完时就超时。
    """
    zf = F.tele.z_filtered()
    h = max(0.0, (zf - F.INITIAL_Z)) if zf is not None else C.MAX_HEIGHT
    return max(C.AUTO_LAND_TIMEOUT, h / C.AUTO_LAND_SPEED_EST * 1.5 + 10.0)


def phase_auto_land(set_mode):
    px, py = F.tele.xy()
    horiz_now = math.hypot(px - F.INITIAL_X, py - F.INITIAL_Y)
    horiz_lim = max(C.PHASE_ENVELOPE["auto_land"][0], horiz_now + C.ENVELOPE_MARGIN)
    set_phase("auto_land", horiz=horiz_lim)

    # 被接管了就不能再切模式 —— 那是在抢飞手的控制权
    if F.takeover[0] is not None:
        rospy.logwarn(f"[auto_land] 控制权在 {F.takeover[0]} — 不切 AUTO.LAND")
        return False

    F.voluntary_land[0] = True  # 必须先设 flag 再切 mode
    rospy.loginfo("[auto_land] 切 AUTO.LAND")
    try:
        r = set_mode(custom_mode="AUTO.LAND")
        if not r.mode_sent:
            rospy.logwarn("[auto_land] SetMode not sent")
            return False
    except Exception as e:
        rospy.logerr(f"[auto_land] exception: {e}")
        return False

    tmo = auto_land_timeout()
    rospy.loginfo(f"[auto_land] 等待 PX4 落地上锁 (最长 {tmo:.0f}s)")
    t_end = rospy.Time.now() + rospy.Duration(tmo)
    with _safety_keys("auto_land") as keys:
        while rospy.Time.now() < t_end:
            keys, _ = _poll_safety_keys(keys, "auto_land", allow_land=False)
            if F.abort_flag.is_set():
                return False
            if F.takeover[0] is not None:
                # AUTO.LAND 之后飞手又切走了 (safety_monitor 现在也看这种情况)
                rospy.logwarn(f"[auto_land] 降落中控制权被接管 ({F.takeover[0]}) — 停手")
                return False
            if F.tele.state and not F.tele.state.armed:
                rospy.loginfo("[auto_land] PX4 已在着陆时 disarm")
                F.in_air[0] = False
                return True
            rospy.sleep(0.05)
    rospy.logwarn("[auto_land] TIMEOUT — PX4 未在窗口内 disarm "
                  "(不会因此停桨: cleanup 看到仍是 AUTO.LAND 会让 PX4 自己降完)")
    return False


def phase_handover():
    """
    控制权被接管后的唯一阶段。

    脚本在这里**什么都不主动做**:
      · 不切模式, 不停桨
      · 飞控不在 OFFBOARD 时, 把设定点持续更新成"当前位置" —— 这样如果飞手
        又切回 OFFBOARD, 飞机原地悬停, 而不是冲向一个几秒前的旧设定点;
        回到 OFFBOARD 后设定点冻结在那一刻 (不跟着漂)
      · 等飞机上锁 (飞手/PX4 已经降落) 后正常退出
    键盘: 只有 X 有效 (人工停桨)。Ctrl+C 退出脚本**不动电机**。
    """
    set_phase("handover", horiz=1e9, vel=1e9)   # 包络不再适用: 飞机不归脚本管
    F.sub_mode_ref[0] = ""
    rospy.logerr("=" * 64)
    rospy.logerr(f"[handover] 控制权在 {F.takeover[0]} —— 脚本只看不动, 等飞机落地上锁")
    rospy.logerr("[handover] X = 紧急停桨  |  Ctrl+C = 退出脚本(不动电机)")
    rospy.logerr("[handover] 若飞手切回 OFFBOARD: 飞机原地悬停, 用 QGC 切 Land 降落")
    rospy.logerr("=" * 64)

    keys = None
    if sys.stdin.isatty():
        try:
            keys = KeyReader()
            keys.__enter__()
        except Exception:
            keys = None
    rate = rospy.Rate(C.PHASE_LOOP_HZ)
    t_log = rospy.Time.now()
    try:
        while not rospy.is_shutdown():
            check_abort()
            st = F.tele.state
            if st is not None and not st.armed:
                F.in_air[0] = False
                rospy.loginfo(f"[handover] 飞机已上锁 (模式 {st.mode}) — 退出")
                return
            if keys is not None:
                _, flags, eof, _ = keys.poll()
                if 'x' in flags:
                    request_abort("handover X 键")
                    check_abort()
                # EOF 在这里没有动作可做: 降落本来就不归脚本管
            if st is not None and st.mode != "OFFBOARD" and F.tele.pose is not None:
                p = F.tele.pose.pose.position
                F.setpoint.set_zhold_mode(p.x, p.y, p.z, F.tele.yaw_measured(), 0.0, 0.0)
            now = rospy.Time.now()
            if (now - t_log).to_sec() >= 3.0:
                zf = F.tele.z_filtered()
                dz = (zf - F.INITIAL_Z) if zf is not None else float('nan')
                rospy.logwarn(f"[handover] 模式 {st.mode if st else '?'}  dz={dz:+.2f}  "
                              f"等待落地上锁…")
                t_log = now
            rate.sleep()
    finally:
        if keys is not None:
            keys.__exit__(None, None, None)


def on_ground():
    """
    cleanup 用的"飞机是不是已经在地上"。
    优先信 PX4 的 landed_state; 读不到时才退回看 VIO 高度
    (只看 VIO 高度不可靠: 它在空中也可能往下漂 —— z_drop_events.log 就是为这个记的)。
    """
    if not F.in_air[0]:
        return True                       # 从没检测到离地
    _, ext = F.tele.snap_ekf()
    ls = getattr(ext, "landed_state", 0) if ext is not None else 0
    if ls == 1:                           # LANDED_STATE_ON_GROUND
        return True
    if ls in (2, 3, 4):                   # IN_AIR / TAKEOFF / LANDING
        return False
    zf = F.tele.z_filtered()
    return zf is not None and (zf - F.INITIAL_Z) < C.TAKEOFF_LIFTOFF_DZ


def cleanup_decision(fatal, takeover_mode, armed, grounded, mode):
    """
    脚本退出时对电机做什么 (纯函数)。

    返回:
      "none"      已上锁, 什么都不用做
      "disarm"    force_disarm (停桨)
      "leave"     什么都不做, 让当前控制方(飞手 / PX4 的 AUTO.LAND)继续
      "auto_land" 请求 AUTO.LAND, 不停桨

    空中停桨只有一个来源: FATAL (姿态失控 / MAVROS 断链 / 操作员按 X)。
    代码异常、Ctrl+C、AUTO.LAND 等太久、飞手接管都不停桨。
    """
    if not armed:
        return "none"
    if fatal:
        return "disarm"
    if grounded:
        return "disarm"                   # 地上: 停桨无害, 也是必须的
    if takeover_mode is not None:
        return "leave"                    # 飞手/PX4 在控制, 不碰
    if mode == "AUTO.LAND":
        return "leave"                    # PX4 正在自己降, 让它降完
    return "auto_land"


def restore_ground_mode(set_mode, grounded):
    """
    退出前整理飞控模式, 让下一次运行从干净状态开始。

    只在 "地上 + 已上锁 + 没被接管 + MAVROS 在线" 时做 —— 空中切模式会改变飞行,
    被接管时切模式是抢飞手的控制权。不满足就什么都不做。
    2026-10-19 (首飞问题 §5): 只有停在 **OFFBOARD** 才需要切走 (停发设定点后地上会触发
    offboard 丢失保护)。正常降落后是 AUTO.LAND, 保持不动 —— 室内没遥控、没全局位置时
    PX4 本来就不接受 MANUAL / AUTO.LOITER, 以前每次都白报两条警告。
    OFFBOARD 时依次试: 开局模式 → PARK_MODE → AUTO.LAND (PX4 只要有本地位置就接受)。
    返回最终模式 (没动则 None)。
    """
    st = F.tele.state
    if st is None or not st.connected or st.armed or not grounded:
        return None
    if F.takeover[0] is not None:
        return None
    if st.mode != "OFFBOARD":
        rospy.loginfo(f"cleanup: 模式 {st.mode}, 保持不动")
        return st.mode
    tried = []
    for m in (F.start_mode[0], C.PARK_MODE, "AUTO.LAND"):
        if not m or m == "OFFBOARD" or m in tried:
            continue
        tried.append(m)
        try:
            r = set_mode(custom_mode=m)
            if not r.mode_sent:
                rospy.loginfo(f"cleanup: 切 {m} 没发出去")
                continue
        except Exception as e:
            rospy.loginfo(f"cleanup: 切 {m} 异常: {e}")
            continue
        t_end = time.time() + 1.0          # time.time: 不依赖 ROS 时钟
        while time.time() < t_end:
            if F.tele.state and F.tele.state.mode == m:
                rospy.loginfo(f"cleanup: 模式 OFFBOARD → {m}")
                return m
            time.sleep(0.05)
        rospy.loginfo(f"cleanup: PX4 没接受 {m}")
    cur = F.tele.state.mode if F.tele.state else "?"
    rospy.logwarn(f"cleanup: 停在 {cur} 切不走 (试过 {', '.join(tried)}) —— 下一次开局会先切 {C.PARK_MODE}")
    return None


def arm_with_confirm(arm_srv):
    """
    解锁。被拒时打印 MAV_RESULT + 飞控 STATUSTEXT 原文, 然后**等操作员回车确认**再重试。

    不自动重试: 被拒后现场的人可能已经走到飞机旁边看情况, 这时自动重试一旦成功,
    飞机会立刻解锁并直接进入起飞。等待确认期间 setpoint 照常发布, 模式保持 OFFBOARD。
    """
    for attempt in range(1, C.ARM_MAX_TRIES + 1):
        check_abort()
        cur = F.tele.state.mode if F.tele.state else "?"
        if cur != "OFFBOARD":
            raise RuntimeError(f"arm 前模式已不是 OFFBOARD (当前 {cur})")

        t0 = rospy.Time.now().to_sec()
        rospy.loginfo(f"[arm] arming (第 {attempt}/{C.ARM_MAX_TRIES} 次)")
        r = arm_srv(True)

        if r.success:
            t_end = rospy.Time.now() + rospy.Duration(C.ARM_TIMEOUT)
            while rospy.Time.now() < t_end:
                check_abort()
                if F.tele.state and F.tele.state.armed:
                    # 正常解锁时 PX4 会发 "Armed by external command"。一条都没有
                    # 说明 STATUSTEXT 没通 —— 下次被拒就看不到原因, 提前喊出来。
                    # 放后台线程里查, 不耽误起飞。
                    def _check_fcu_link(t_ref=t0):
                        if not fcu_texts_since(t_ref - 0.2):
                            rospy.logwarn("[arm] 解锁后 2s 内没收到任何飞控消息 —— STATUSTEXT "
                                          "链路不通, 下次被拒时看不到原因 "
                                          "(查 rostopic echo /mavros/statustext/recv)")
                    _tm = threading.Timer(2.0, _check_fcu_link)
                    _tm.daemon = True
                    _tm.start()
                    return
                rospy.sleep(0.05)
            raise RuntimeError("arm not confirmed")

        # STATUSTEXT 通常比 COMMAND_ACK 晚几十 ms 到, 等一下再取
        rospy.sleep(1.0)
        res = getattr(r, "result", -1)
        rospy.logerr(f"[arm] 第 {attempt} 次被拒  MAV_RESULT={res} "
                     f"({_MAV_RESULT.get(res, '?')})")
        why = fcu_texts_since(t0 - 0.2)
        if why:
            for txt in why:
                rospy.logerr(f"[arm]   飞控: {txt}")
        else:
            rospy.logerr("[arm]   没收到任何飞控消息 —— STATUSTEXT 链路不通, 看不到拒绝原因。"
                         "看 QGC 消息栏, 或在 QGC MAVLink 控制台跑 commander check")

        if attempt >= C.ARM_MAX_TRIES:
            break
        _flush_stdin()
        ans = input(f"\n[arm] 确认桨周围无人后按回车重试 ({attempt + 1}/{C.ARM_MAX_TRIES}), "
                    f"输入 q 回车退出: ").strip().lower()
        if ans == "q":
            break

    raise RuntimeError("arm rejected")


def single_instance_guard():
    """
    已有飞控脚本在跑就拒绝启动。
      · 同名 /flight_ctrl: ROS master 会直接关掉**正在飞的那个** (它的 cleanup 也发不出指令)
      · 旧版 /stage9_track: 不会被关, 两个脚本同时往飞控发设定点
    master 里登记着但进程已死 (上次被强杀) → 只警告, 新节点会顶掉旧登记。
    """
    try:
        import rosgraph
        import socket
        import xmlrpc.client
        master = rosgraph.Master("/flight_ctrl_precheck")
        for name in ("/" + C.NODE_NAME, "/stage9_track"):
            try:
                uri = master.lookupNode(name)
            except rosgraph.MasterError:
                continue                   # 没登记
            old = socket.getdefaulttimeout()
            socket.setdefaulttimeout(1.0)
            try:
                xmlrpc.client.ServerProxy(uri).getPid("/flight_ctrl_precheck")
                alive = True
            except Exception:
                alive = False
            finally:
                socket.setdefaulttimeout(old)
            if alive:
                print("\n" + "!" * 78)
                print(f"拒绝启动: 飞控脚本 {name} 正在运行 ({uri})")
                print("  同名节点会把正在飞的那个直接关掉; 旧版脚本则会和本脚本同时发设定点。")
                print("  先结束那个脚本 (它的终端里按 L 降落 / Ctrl+C), 再启动。")
                print("!" * 78 + "\n")
                sys.exit(1)
            print(f"[guard] {name} 在 master 里有登记但进程已不在 (上次被强杀?) —— 继续")
    except SystemExit:
        raise
    except Exception as e:
        # roscore 都连不上时 init_node 自己会卡住/报错, 这里不替它拦
        print(f"[guard] 查重失败 (跳过检查): {e}")


def emergency_land(set_mode, z_now, yaw_hold):
    """
    LAND 级 abort 的降落路径: 就地冻结 xy → 缓降 → AUTO.LAND。
    不回原点 —— 触发 LAND 的原因往往正是"位置/速度不可信"。
    """
    px, py = F.tele.xy()
    rospy.logwarn(f"[emergency_land] 就地降落 @ ({px:+.2f},{py:+.2f})")
    try:
        phase_descend(z_now, yaw_hold, px, py)
    except Exception as e:
        rospy.logerr(f"[emergency_land] descend 异常: {e}")
    if F.abort_flag.is_set():
        return False
    return phase_auto_land(set_mode)


# =============================================================================
# ==========================   Pose callback   ================================
# =============================================================================
def pose_cb(msg):
    """
    回调里只做两件极快的事: 更新 telemetry、往队列塞一个元组。
    ROS 单订阅者的回调是串行的, 这里一次几百 ms 的文件 IO 就会把下一帧
    pose 顶到 POSE_STALE_SEC 之外。timing.csv 由后台线程落盘。
    """
    prev_stamp = F.tele.pose_stamp_prev
    F.tele.set_pose(msg)
    try:
        now = rospy.Time.now().to_sec()
        stamp = msg.header.stamp.to_sec()
        if prev_stamp is not None and prev_stamp.to_sec() > 0:
            interval_ms = (stamp - prev_stamp.to_sec()) * 1000.0
        else:
            interval_ms = float('nan')
        with timing_queue_lock:
            timing_queue.append((now, stamp, (now - stamp) * 1000.0,
                                 interval_ms, F.tele.stamp_backward_count,
                                 F.tele.last_jump))
    except Exception:
        pass
