import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import json
import torch
import torchvision
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
#123
USE_GPIO = os.getenv("ENABLE_GPIO", "0") == "1"
GPIO = None
HAS_GPIO = False
if USE_GPIO:
    try:
        import Jetson.GPIO as GPIO
        HAS_GPIO = True
    except ImportError:
        pass

PIN_IN = 22
PIN_OUTS = [16]

try:
    from vlm_harness import DetectionSummary, build_vlm_harness
except Exception:
    DetectionSummary = None
    build_vlm_harness = None

#123
# Load devices from JSON config
def load_devices() -> List[dict]:
    config_path = os.path.join(os.path.dirname(__file__), "devices.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return [d for d in config.get("devices", []) if d.get("enabled", True)]
    except FileNotFoundError:
        # Fallback to single device if config doesn't exist
        return [{"id": "0024E979D6", "name": "預設路燈", "ip": "192.168.50.202", "enabled": True}]


def save_devices(devices: List[dict]) -> None:
    config_path = os.path.join(os.path.dirname(__file__), "devices.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"devices": devices}, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log(f"save-devices-failed: {exc}")


DEVICES = load_devices()
LIGHT_TARGET_IP = os.getenv("LIGHT_TARGET_IP", DEVICES[0]["ip"])
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "5050"))
TCP_HOST = os.getenv("TCP_HOST", "0.0.0.0")
TCP_PORT = int(os.getenv("TCP_PORT", "5566"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "960"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "540"))
CAMERA_FPS = float(os.getenv("CAMERA_FPS", "30"))
CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "MJPG").upper()
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
INFER_IMGSZ = int(os.getenv("INFER_IMGSZ", "640"))
YOLO_MODEL = os.getenv(
    "YOLO_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo26n.pt"),
)
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")


def _torch_nms_fallback(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep).to(dtype=torch.long)


try:
    torchvision.ops.nms(
        torch.zeros((0, 4), device=YOLO_DEVICE),
        torch.zeros((0,), device=YOLO_DEVICE),
        0.45,
    )
except Exception as exc:
    torchvision.ops.nms = _torch_nms_fallback
    print(f"[startup] torchvision.ops.nms fallback enabled: {exc}", flush=True)

app = Flask(__name__)
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)


@dataclass
class LightLevel:
    level: int
    label: str
    brightness: int


AUTO_LEVELS = [
    LightLevel(0, "standby", 0),
    LightLevel(1, "low_pedestrian", 10),
    LightLevel(2, "mid_pedestrian", 30),
    LightLevel(3, "high_pedestrian", 60),
    LightLevel(4, "emergency", 100),
]
DAYLIGHT_LUX_THRESHOLD = 270
MID_PEOPLE_MAX = 3
#倒數秒數

COUNTDOWN_SECONDS = int(os.getenv("COUNTDOWN_SECONDS", "15"))


state_lock = threading.RLock()
state = {
    "mode": "manual",
    "person_count": 0,
    "level": AUTO_LEVELS[0].level,
    "level_label": AUTO_LEVELS[0].label,
    "brightness": AUTO_LEVELS[0].brightness,
    "device_connected": False,
    "device_ip": LIGHT_TARGET_IP,
    "devices": DEVICES,
    "device_brightness": {d["id"]: None for d in DEVICES},
    "device_brightness_ts": {},
    "device_lost": {d["id"]: False for d in DEVICES},
    "device_detail": {d["id"]: {} for d in DEVICES},
    "tcp_error": "",
    "last_packet_hex": "",
    "vlm_enabled": os.getenv("ENABLE_VLM", "0") == "1",
    "vlm_busy": False,
    "vlm_last_alert": None,
    "vlm_last_trigger": None,
    "vlm_harness_status": "disabled",
    "yolo_device": YOLO_DEVICE,
    "torch_cuda": torch.cuda.is_available(),
    "countdown_active": False,
    "countdown_remaining": 0,
    "lux_avg": None,
    "lux_raw": None,
    "lux_last_seen": None,
    "lux_failures": 0,
    "lux_error": "",
    "pending_level": None,
    "pending_brightness": None,
    "device_target_brightness": {d["id"]: None for d in DEVICES},
    "device_last_command_time": {d["id"]: 0.0 for d in DEVICES},
    "camera_fps": 0.0,
    "pipeline_fps": 0.0,
    "infer_ms": 0.0,
    "capture_ms": 0.0,
    "encode_ms": 0.0,
    "camera_fourcc": CAMERA_FOURCC,
    "camera_requested_fps": CAMERA_FPS,
    "camera_status": "starting",
    "camera_image_status": "unknown",
    "camera_mean_luma": None,
    "yolo_status": "loading",
    "yolo_error": "",
    "yolo_last_success": None,
    "emergency_trigger": False,
    "gpio_status": "starting" if USE_GPIO else "disabled",
    "telemetry_history": [],
}

# Countdown timer control
_countdown_cancel = threading.Event()
_countdown_generation = 0
_retry_ready = threading.Condition()
_retry_inflight: Dict[Tuple[int, str], dict] = {}
RETRY_DELAYS = (2.0, 2.0, 5.0, 5.0, 5.0, 5.0)
GROUP_FOLLOWUP_DELAYS = (2.0, 3.0, 3.0)
GROUP_CONFIRMATION_GRACE = 1.0
COMMAND_TIMEOUT = 30.0
AUTO_LEVEL_STABLE_SEC = float(os.getenv("AUTO_LEVEL_STABLE_SEC", "1.0"))
_auto_candidate_level: Optional[int] = None
_auto_candidate_since = 0.0

active_tcp_clients: Dict[str, socket.socket] = {}
latest_frame_lock = threading.Lock()
latest_jpeg: Optional[bytes] = None
latest_raw_frame = None
logs = deque(maxlen=200)
vlm_harness = build_vlm_harness() if build_vlm_harness else None

# Per-device last-seen timestamp for lost detection
_device_last_seen: Dict[str, float] = {d["id"]: 0.0 for d in DEVICES}
DEVICE_LOST_TIMEOUT = 1000  # 3 minutes

# USB RS-485 Light Sensor (JNC LUX20)
USB_LUX_PORT = os.getenv("USB_LUX_PORT", "/dev/ttyUSB0")
USB_LUX_BAUD = int(os.getenv("USB_LUX_BAUD", "9600"))
USB_LUX_UNIT_ID = int(os.getenv("USB_LUX_UNIT_ID", "77"))
LUX_WINDOW = 5  # rolling-average window in seconds
_lux_samples: deque = deque()  # (timestamp, lux)
_last_lux_error_log = 0.0


def _close_tcp_client(conn: socket.socket) -> None:
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()


def _update_device_connection() -> None:
    with state_lock, _retry_ready:
        state["device_connected"] = any(d["ip"] in active_tcp_clients for d in DEVICES)
    broadcast_state()


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    logs.append(line)
    print(line, flush=True)
    socketio.emit("packet_log", {"msg": line})

#等級切換

def choose_auto_level(person_count: int, lux: Optional[float]) -> LightLevel:
    """Choose the appropriate light level based on environmental lux and person count."""
    if lux is not None and lux > DAYLIGHT_LUX_THRESHOLD:
        return AUTO_LEVELS[0]
    
    # When lux is unavailable, use person count alone.
    if person_count <= 0:
        return AUTO_LEVELS[1] # L1 (low)
    elif person_count <= MID_PEOPLE_MAX:
        return AUTO_LEVELS[2] # L2 (mid)
    else:
        return AUTO_LEVELS[3] # L3 (high)


def build_light_packet(device_id: str, brightness: int) -> bytes:
    """Build a light control packet for a specific device (0x52)."""
    dim_value = max(0, min(100, int(brightness)))
    id_part = device_id[2:]  # Skip first 2 chars
    device_bytes = [int(id_part[i:i+2], 16) for i in range(0, len(id_part), 2)]
    packet = bytearray([0x02, 0x0B, 0x52, 0x00] + device_bytes + [0x01])
    packet.append(dim_value)
    packet.append(sum(packet) % 256)
    return bytes(packet)


def build_group_set_packet(device_id: str, group_id: int) -> bytes:
    """Build a group assignment packet (0x56)."""
    id_part = device_id[2:]
    device_bytes = [int(id_part[i:i+2], 16) for i in range(0, len(id_part), 2)]
    # 02 0B 56 00 [MAC 4 bytes] 01 [Group ID] [CheckSum]
    packet = bytearray([0x02, 0x0B, 0x56, 0x00] + device_bytes + [0x01])
    packet.append(max(0, min(255, int(group_id))))
    packet.append(sum(packet) % 256)
    return bytes(packet)


def build_group_control_packet(device_id: str, group_id: int, brightness: int) -> bytes:
    """Build a group control packet (0x36). Length 12.
    Format: 02 0C 36 [MAC 5 bytes] 01 [Group ID] [Dim] [CheckSum]
    """
    # Use full 5-byte MAC (10 hex chars)
    device_bytes = [int(device_id[i:i+2], 16) for i in range(0, len(device_id), 2)]
    dim_value = max(0, min(100, int(brightness)))
    
    packet = bytearray([0x02, 0x0C, 0x36])
    packet.extend(device_bytes)      # 5 bytes MAC
    packet.append(0x01)              # 01
    packet.append(max(0, min(255, int(group_id)))) # Group ID
    packet.append(dim_value)         # Dimming
    packet.append(sum(packet) % 256) # Checksum
    return bytes(packet)


def send_light(brightness: int, reason: str, target_device: Optional[dict] = None, group_id: Optional[int] = None) -> Tuple[bool, str]:
    """Send light control (0x52) or group control (0x36) command."""
    with state_lock:
        if state["emergency_trigger"] and not (
            brightness == 100 and reason in ("emergency-on", "consistency-sync")
        ):
            return False, "Emergency is active"
        if group_id is not None:
            members = [device for device in DEVICES if device.get("group_id") == group_id]
            if not members:
                return False, f"No devices found in group {group_id}"
            reason = f"{reason}-group-{group_id}"
            representative = members[0]
            now = time.time()
            for device in members:
                state["device_target_brightness"][device["id"]] = brightness
                state["device_last_command_time"][device["id"]] = now
            packet = build_group_control_packet(representative["id"], group_id, brightness)
            enabled = [device for device in members if device.get("enabled", True)]
            _start_retry_worker(representative, packet, reason, enabled or [representative])
            return True, f"Group {group_id} control started via {representative['name']}"

        devices_to_control = [target_device] if target_device else DEVICES

        for device in devices_to_control:
            if not device.get("enabled", True):
                continue

            state["device_target_brightness"][device["id"]] = brightness
            state["device_last_command_time"][device["id"]] = time.time()

            packet = build_light_packet(device["id"], brightness)
            _start_retry_worker(device, packet, reason)

        return True, "Control sequence started"


def _start_retry_worker(device: dict, packet: bytes, reason: str, members=None) -> None:
    opcode = packet[2]
    target_ip = device["ip"]
    devices = {d["id"]: d for d in (members or [device])}
    key = (opcode, str(packet[9]) if opcode == 0x36 else device["id"])
    now = time.monotonic()
    with _retry_ready:
        current = _retry_inflight.get(key)
        if current and current["packet"] == packet and current["original_ids"] == devices.keys():
            return
        if opcode in (0x36, 0x52):
            for other_key, item in list(_retry_inflight.items()):
                if other_key == key or other_key[0] not in (0x36, 0x52):
                    continue
                superseded = item["devices"].keys() & devices.keys()
                for device_id in superseded:
                    del item["devices"][device_id]
                if not item["devices"]:
                    del _retry_inflight[other_key]
        _retry_inflight[key] = {
            "device": device,
            "devices": devices,
            "original_ids": set(devices),
            "awaiting": {},
            "packet": packet,
            "reason": reason,
            "queued_at": now,
            "attempts": 0,
            "retry_delays": RETRY_DELAYS,
            "next_send_time": now,
        }
        connected = target_ip in active_tcp_clients
        _retry_ready.notify()
    if not connected:
        log(f"control-queued-offline: {device['name']} target={target_ip} reason={reason}")


def _confirm_control(device_id: str, cmd: int, value=None) -> None:
    now = time.monotonic()
    with _retry_ready:
        for key, item in list(_retry_inflight.items()):
            sent = item["awaiting"].get(device_id)
            if device_id not in item["devices"] or sent is None:
                continue
            sent_commands, sent_at = sent
            if key[0] == 0x56:
                if cmd not in (0x41, 0x57):
                    continue
            elif value != item["packet"][-2] or (cmd not in (0x40, 0x80) and cmd - 1 not in sent_commands):
                continue
            device = item["devices"].pop(device_id)
            if not item["devices"]:
                del _retry_inflight[key]
            elif key[0] == 0x36 and item["retry_delays"] is RETRY_DELAYS:
                item["attempts"] = 0
                item["retry_delays"] = GROUP_FOLLOWUP_DELAYS
                item["next_send_time"] = now + GROUP_CONFIRMATION_GRACE
            _retry_ready.notify()
            break
        else:
            return
    log(f"control-confirmed: {device['name']} cmd=0x{cmd:02X} value={value} "
        f"since_send_ms={(now - sent_at) * 1000:.1f} total_ms={(now - item['queued_at']) * 1000:.1f}")


def _reconcile_status(device: dict, brightness: int, now: float) -> None:
    device_id = device["id"]
    with state_lock:
        target = state["device_target_brightness"].get(device_id)
        last_cmd = state["device_last_command_time"].get(device_id, 0.0)
        if target is None or brightness == target or now - last_cmd <= 10.0:
            return
        with _retry_ready:
            if any(key[0] in (0x36, 0x52) and device_id in item["devices"]
                   for key, item in _retry_inflight.items()):
                return
        log(f"consistency-sync: {device['name']} reported={brightness} target={target}. Re-sending...")
        send_light(target, "consistency-sync", target_device=device)


def get_public_state() -> dict:
    with state_lock:
        public = dict(state)
    with _retry_ready:
        public["clients"] = sorted(active_tcp_clients)
    public["logs"] = list(logs)[-50:]
    public["server_info"] = f"{TCP_HOST}:{TCP_PORT}"
    return public


def broadcast_state() -> None:
    socketio.emit("state", get_public_state())


def _countdown_worker(target: LightLevel, generation: int) -> None:
    """Generic countdown: waits 30s for conditions to stay stable before downgrading."""
    remaining = COUNTDOWN_SECONDS
    log(f"countdown-start target=L{target.level} {target.label} seconds={COUNTDOWN_SECONDS}")
    
    while remaining > 0:
        with state_lock:
            state["countdown_remaining"] = remaining
            # Check if our target changed or was cancelled during the wait
            if (
                not state["countdown_active"]
                or state["pending_level"] != target.level
                or generation != _countdown_generation
            ):
                # Notify frontend that countdown was cancelled
                socketio.emit("countdown_tick", {"active": False, "remaining": 0})
                return # Exit worker
        # Lightweight per-second tick — avoids full DOM re-render on client
        socketio.emit("countdown_tick", {"active": True, "remaining": remaining})
        time.sleep(1.0)
        remaining -= 1

    # Timer finished successfully – apply the downgrade
    with state_lock:
        if generation != _countdown_generation:
            return
        state["countdown_active"] = False
        state["countdown_remaining"] = 0
        state["pending_level"] = None
        state["level"] = target.level
        state["level_label"] = target.label
        if state["mode"] == "auto":
            state["brightness"] = target.brightness
        else:
            broadcast_state()
            return
            
    log(f"countdown-finished apply-L{target.level} {target.label} brightness={target.brightness}")
    send_light(target.brightness, f"downgrade-to {target.label}", group_id=6)
    broadcast_state()


def set_level_from_count(person_count: int) -> None:
    global _auto_candidate_level, _auto_candidate_since, _countdown_generation
    with state_lock:
        state["person_count"] = person_count
        if state.get("emergency_trigger"):
            _auto_candidate_level = None
            return
        if state["mode"] != "auto":
            _auto_candidate_level = None
            if state["countdown_active"] or state["pending_level"] is not None:
                _countdown_generation += 1
                state["countdown_active"] = False
                state["pending_level"] = None
                state["countdown_remaining"] = 0
                broadcast_state()
            return

        target_level = choose_auto_level(person_count, state.get("lux_avg"))
        current_level_idx = state["level"]
        now = time.monotonic()

        # Confirm an upgrade after the configured stability period.
        if target_level.level > current_level_idx:
            # Require stable candidate for upgrades to avoid flickering
            if _auto_candidate_level != target_level.level:
                _auto_candidate_level = target_level.level
                _auto_candidate_since = now
            if now - _auto_candidate_since < AUTO_LEVEL_STABLE_SEC:
                return
            _countdown_generation += 1
            state["countdown_active"] = False
            state["pending_level"] = None
            state["countdown_remaining"] = 0
            state["level"] = target_level.level
            state["level_label"] = target_level.label
            state["brightness"] = target_level.brightness
            _auto_candidate_level = None  # reset after upgrade
            log(f"auto-upgrade to L{target_level.level} {target_level.label} (immediate)")
            send_light(target_level.brightness, f"upgrade-L{target_level.level}", group_id=6)
            broadcast_state()
            return

        # 2. EQUAL Logic (target == current) -> Cancel any pending downgrade
        if target_level.level == current_level_idx:
            _auto_candidate_level = None  # reset candidate when stable
            if state["countdown_active"]:
                _countdown_generation += 1
                state["countdown_active"] = False
                state["pending_level"] = None
                state["countdown_remaining"] = 0
                log("countdown-cancelled: conditions returned to current level")
                broadcast_state()
            return

        # 3. DOWNGRADE Logic (target < current) -> 30s Confirmation (no candidate delay)
        if target_level.level < current_level_idx:
            _auto_candidate_level = None  # reset candidate on downgrade path
            if state["pending_level"] != target_level.level:
                # New target or restarted condition
                _countdown_generation += 1
                state["countdown_active"] = True
                state["pending_level"] = target_level.level
                state["countdown_remaining"] = COUNTDOWN_SECONDS
                generation = _countdown_generation
                threading.Thread(target=_countdown_worker, args=(target_level, generation), daemon=True).start()
                log(f"countdown-start: waiting {COUNTDOWN_SECONDS}s to downgrade to L{target_level.level}")
                broadcast_state()
            return


def start_tcp_server() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((TCP_HOST, TCP_PORT))
            server.listen()
            with state_lock:
                state["tcp_error"] = ""
            log(f"tcp-server started host={TCP_HOST} port={TCP_PORT}")
            while True:
                conn, addr = server.accept()
                client_ip = addr[0]
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(1.0)
                with _retry_ready:
                    previous = active_tcp_clients.get(client_ip)
                    active_tcp_clients[client_ip] = conn
                    now = time.monotonic()
                    for item in _retry_inflight.values():
                        if any(d["ip"] == client_ip for d in item["devices"].values()):
                            if item["attempts"] < len(item["retry_delays"]):
                                item["next_send_time"] = now
                    _retry_ready.notify()
                if previous is not None:
                    _close_tcp_client(previous)
                log(f"device-connected {client_ip}:{addr[1]}")
                _update_device_connection()
                threading.Thread(target=handle_tcp_client, args=(conn, client_ip), daemon=True).start()
    except OSError as exc:
        message = f"tcp-server failed host={TCP_HOST} port={TCP_PORT}: {exc}"
        with state_lock:
            state["tcp_error"] = message
        log(message)
        broadcast_state()


def _append_history(device_name: str, packet_hex: str, cmd_hex: str, brightness: int, now: float,
                    voltage=None, current=None, active_p=None, pf=None, rssi=None):
    hist_entry = {
        "time": time.strftime("%H:%M:%S", time.localtime(now)),
        "cmd": cmd_hex,
        "name": device_name,
        "dimming": brightness,
        "voltage": voltage,
        "current": current,
        "active_p": active_p,
        "pf": pf,
        "rssi": rssi,
        "raw_hex": packet_hex
    }
    with state_lock:
        history = list(state.get("telemetry_history", []))
        history.append(hist_entry)
        if len(history) > 20:
            history = history[-20:]
        state["telemetry_history"] = history

def _update_reported_brightness(device: dict, brightness: int, cmd: int, now: float,
                                detail: Optional[dict] = None) -> bool:
    if not 0 <= brightness <= 100:
        log(f"device-rx-invalid-brightness device={device['id']} brightness={brightness}")
        return False
    dev_id = device["id"]
    with state_lock:
        updated = dict(state["device_detail"].get(dev_id, {}))
        if detail is not None:
            updated.update(detail)
        updated.update({
            "device_id": dev_id,
            "name": device["name"],
            "mac": dev_id,
            "cmd": f"0x{cmd:02X}",
            "dimming": brightness,
            "ts": now,
            "telemetry_stale": detail is None,
        })
        state["device_brightness"][dev_id] = brightness
        state["device_brightness_ts"][dev_id] = now
        state["device_detail"][dev_id] = updated
    socketio.emit("device_brightness_update", {
        "device_id": dev_id, "mac": dev_id, "brightness": brightness,
        "cmd": f"0x{cmd:02X}", "ts": now,
    })
    socketio.emit("device_detail_update", updated)
    return True


def _parse_status_packet(packet: bytes, client_ip: str) -> None:
    """Parse incoming packets: 0x40/0x80 status and 0x41/0x53/0x37 ACK packets."""
    if len(packet) < 3:
        return
    cmd = packet[2]
    packet_hex = packet.hex().upper()
    with state_lock:
        state["last_packet_hex"] = packet_hex
    
    # Common MAC extraction for 0x80, 0x53, 0x37 (bytes 4-8, indices 3-7)
    if len(packet) < 8:
        log(f"device-rx-short ip={client_ip} hex={packet_hex}")
        return
    mac_bytes = packet[3:8]
    device_id = mac_bytes.hex().upper()
    packet_hex = packet.hex().upper()
    
    # Match to a known device
    matched_device = None
    for d in DEVICES:
        if d["id"].upper() == device_id.upper():
            matched_device = d
            break
    if not matched_device:
        log(f"device-rx-unmatched ip={client_ip} cmd=0x{cmd:02X} device={device_id} hex={packet_hex}")
        return

    dev_id = matched_device["id"]
    now = time.time()
    _device_last_seen[dev_id] = now
    
    with state_lock:
        state["device_lost"][dev_id] = False
        
    if cmd == 0x80 and len(packet) >= 28:
        # Full telemetry parse
        serial      = packet[8]
        rssi        = packet[9]
        voltage     = round(((packet[10] << 8) | packet[11]) / 10.0, 1)   # V
        current     = round(((packet[12] << 8) | packet[13]) / 100.0, 2)  # A
        freq        = packet[14]                                            # Hz
        pf          = round(((packet[15] << 8) | packet[16]) / 1000.0, 3) # 0-1
        active_p    = round(((packet[17] << 8) | packet[18]) / 10.0, 1)   # W
        apparent_p  = round(((packet[19] << 8) | packet[20]) / 10.0, 1)   # VA
        reactive_p  = round(((packet[21] << 8) | packet[22]) / 10.0, 1)   # VAR
        brightness  = packet[23]                                            # %
        group       = packet[24]
        working_min = (packet[25] << 8) | packet[26]                       # minutes

        detail = {
            "device_id":    dev_id,
            "name":         matched_device["name"],
            "mac":          device_id,
            "cmd":          f"0x{cmd:02X}",
            "serial":       serial,
            "rssi":         rssi,
            "voltage":      voltage,
            "current":      current,
            "freq":         freq,
            "pf":           pf,
            "active_p":     active_p,
            "apparent_p":   apparent_p,
            "reactive_p":   reactive_p,
            "dimming":      brightness,
            "group":        group,
            "working_min":  working_min,
            "ts":           now,
            "telemetry_stale": False,  # Fresh 0x80 data — not stale
        }
        if not _update_reported_brightness(matched_device, brightness, cmd, now, detail):
            return
        # Add to telemetry history
        _append_history(matched_device["name"], packet_hex, f"0x{cmd:02X}", brightness, now,
                        voltage, current, active_p, pf, rssi)

        log(f"device-status ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} brightness={brightness}")
        _confirm_control(dev_id, cmd, brightness)
        _reconcile_status(matched_device, brightness, now)

    elif cmd == 0x40 and len(packet) >= 14:
        brightness = packet[13]
        if not _update_reported_brightness(matched_device, brightness, cmd, now):
            return
        _append_history(matched_device["name"], packet_hex, f"0x{cmd:02X}", brightness, now)
        log(f"device-status ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} brightness={brightness}")
        _confirm_control(dev_id, cmd, brightness)
        _reconcile_status(matched_device, brightness, now)

    elif cmd in (0x41, 0x53, 0x37):
        if len(packet) >= 14:
            # For 0x41, 0x53, 0x37, brightness is at index 13
            brightness = packet[13]
            if not _update_reported_brightness(matched_device, brightness, cmd, now):
                return
            _append_history(matched_device["name"], packet_hex, f"0x{cmd:02X}", brightness, now)
            log(f"device-ack-update ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} brightness={brightness}%")
            _confirm_control(dev_id, cmd, brightness)
        else:
            log(f"device-ack ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']}")
            _confirm_control(dev_id, cmd)
    elif cmd == 0x57:
        _confirm_control(dev_id, cmd)
    else:
        log(f"device-rx-unhandled ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} hex={packet_hex}")
        
    broadcast_state()


def _central_retry_worker() -> None:
    while True:
        due = None
        expired = []
        with _retry_ready:
            now = time.monotonic()
            next_wake = None
            for key, item in list(_retry_inflight.items()):
                disconnect_deadline = item["queued_at"] + COMMAND_TIMEOUT
                packet = item["packet"]
                devices = item["devices"]
                if key[0] == 0x36 and devices.keys() != item["original_ids"]:
                    sends = [(d["ip"], build_light_packet(device_id, packet[-2]), (device_id,))
                             for device_id, d in devices.items()]
                else:
                    sends = [(item["device"]["ip"], packet, tuple(devices))]
                connected = any(ip in active_tcp_clients for ip, _, _ in sends)
                deadline = item["next_send_time"] if connected else disconnect_deadline
                retry_delays = item["retry_delays"]
                exhausted = connected and item["attempts"] >= len(retry_delays)
                if deadline <= now:
                    if exhausted or not connected:
                        expired.append(item)
                        del _retry_inflight[key]
                    elif due is None:
                        item["next_send_time"] = now + retry_delays[item["attempts"]]
                        item["attempts"] += 1
                        due = (key, item, sends, item["attempts"])
                elif next_wake is None or deadline < next_wake:
                    next_wake = deadline
            if due is None and not expired:
                timeout = None if next_wake is None else max(0.0, next_wake - now)
                _retry_ready.wait(timeout)
                continue

        for item in expired:
            missing = ", ".join(d["name"] for d in item["devices"].values())
            log(f"retry-exhausted: {missing} did not confirm reason={item['reason']}")

        if due is None:
            continue
        key, item, sends, attempt = due
        for ip, packet, recipients in sends:
            with _retry_ready:
                conn = active_tcp_clients.get(ip)
                if (_retry_inflight.get(key) is not item or conn is None
                        or any(device_id not in item["devices"] for device_id in recipients)):
                    continue
                sent_at = time.monotonic()
                for device_id in recipients:
                    sent_commands, _ = item["awaiting"].get(device_id, (set(), None))
                    item["awaiting"][device_id] = (sent_commands | {packet[2]}, sent_at)
                name = item["device"]["name"] if packet[2] == 0x36 else item["devices"][recipients[0]]["name"]
            try:
                conn.sendall(packet)
            except OSError as exc:
                log(f"tcp-send-error: ip={ip} reason={item['reason']} error={exc}")
                _close_tcp_client(conn)
                continue
            send_ms = (time.monotonic() - sent_at) * 1000
            phase = "p1" if attempt <= 3 else "p2"
            log(f"retry-{phase}: {name} hex={packet.hex().upper()} reason={item['reason']} "
                f"dispatch_ms={(sent_at - item['queued_at']) * 1000:.1f} "
                f"send_ms={send_ms:.1f}")
            if attempt == 1 and packet[2] in (0x36, 0x52):
                _append_history(name, packet.hex().upper(), f"0x{packet[2]:02X}", packet[-2], time.time())


def handle_tcp_client(conn: socket.socket, client_ip: str) -> None:
    buffer = b""
    try:
        with conn:
            while True:
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue
                if not data:
                    break
                with _retry_ready:
                    if active_tcp_clients.get(client_ip) is not conn:
                        break
                buffer += data
                while len(buffer) >= 2:
                    if buffer[0] != 0x02 or buffer[1] < 9:
                        buffer = buffer[1:]
                        continue
                    expected_length = buffer[1]
                    if len(buffer) < expected_length:
                        break
                    packet = buffer[:expected_length]
                    buffer = buffer[expected_length:]
                    
                    # Unified parser for 0x80, 0x53, 0x37
                    _parse_status_packet(packet, client_ip)
    except Exception as exc:
        log(f"device-error ip={client_ip} error={exc}")
    finally:
        with _retry_ready:
            was_current = active_tcp_clients.get(client_ip) is conn
            if was_current:
                del active_tcp_clients[client_ip]
                _retry_ready.notify()
        if was_current:
            log(f"device-disconnected {client_ip}")
            _update_device_connection()


def encode_jpeg(frame) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if ok else b""


class ThreadedCamera:
    def __init__(self, index: int):
        self.capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if CAMERA_FOURCC:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC[:4]))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok = False
        self.frame = None
        self.seq = 0
        self.lock = threading.Lock()
        self.frames = 0
        self.last_fps_check = time.perf_counter()
        self.read_fps = 0.0
        self.running = False
        self.thread = threading.Thread(target=self.update, daemon=True)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.capture.isOpened():
            self.capture.release()

    def start(self) -> "ThreadedCamera":
        self.running = True
        self.thread.start()
        return self

    def describe(self) -> str:
        fourcc_value = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4))
        return (
            f"camera actual={self.capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
            f"{self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}@"
            f"{self.capture.get(cv2.CAP_PROP_FPS):.1f} fourcc={fourcc}"
        )

    def update(self) -> None:
        while self.running:
            grabbed, frame = self.capture.read()
            now = time.perf_counter()
            with self.lock:
                self.ok = grabbed
                if grabbed:
                    self.frame = frame
                    self.seq += 1
                    self.frames += 1
                elapsed = now - self.last_fps_check
                if elapsed >= 2.0:
                    self.read_fps = self.frames / elapsed
                    self.frames = 0
                    self.last_fps_check = now
            if not grabbed:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return self.ok, frame, self.read_fps, self.seq


def boxes_from_result(result, width: int, height: int) -> List[dict]:
    boxes = []
    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        conf = float(box.conf[0].detach().cpu()) if box.conf is not None else 0.0
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        boxes.append(
            {
                "class": "person",
                "confidence": round(conf, 3),
                "xyxy": [round(x1), round(y1), round(x2), round(y2)],
                "width_ratio": round(max(0.0, x2 - x1) / max(1, width), 3),
                "height_ratio": round(max(0.0, y2 - y1) / max(1, height), 3),
            }
        )
    return boxes


def maybe_run_vlm(frame, summary) -> None:
    if vlm_harness is None:
        return

    trigger = vlm_harness.begin(summary)
    with state_lock:
        state["vlm_harness_status"] = vlm_harness.status
    if trigger is None:
        return

    with state_lock:
        state["vlm_busy"] = True
        state["vlm_last_trigger"] = trigger
    broadcast_state()

    jpeg = encode_jpeg(frame)

    def worker() -> None:
        try:
            alert = vlm_harness.inspect(jpeg, summary, trigger)
            with state_lock:
                state["vlm_last_alert"] = alert
            socketio.emit("vlm_alert", alert)
            broadcast_state()
            if alert.get("alert"):
                log(f"vlm-alert risk={alert.get('risk_level', 'unknown')} msg={alert.get('message', '')}")
            else:
                log(f"vlm-clear trigger={trigger.get('reason', 'unknown')}")
        except Exception as exc:
            log(f"vlm-error {exc}")
        finally:
            vlm_harness.finish()
            with state_lock:
                state["vlm_busy"] = False
                state["vlm_harness_status"] = vlm_harness.status
            broadcast_state()

    threading.Thread(target=worker, daemon=True).start()


def load_yolo_model():
    log(
        "loading-yolo "
        f"model={YOLO_MODEL} device={YOLO_DEVICE} "
        f"torch={torch.__version__} cuda={torch.cuda.is_available()}"
    )
    # Let the prediction backend prepare precision and fusion on the target device.
    return YOLO(YOLO_MODEL)


def camera_image_health(frame) -> Tuple[str, float]:
    gray = cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    dark = mean_luma < 8.0 and float((gray < 16).mean()) > 0.98
    return ("dark" if dark else "ok"), round(mean_luma, 2)


def camera_worker() -> None:
    global latest_jpeg, latest_raw_frame, global_camera
    while True:
        with state_lock:
            state["yolo_status"] = "loading"
            state["yolo_error"] = ""
        broadcast_state()
        try:
            _camera_pipeline()
        except Exception as exc:
            with state_lock:
                state["yolo_status"] = "error"
                state["yolo_error"] = f"{type(exc).__name__}: {exc}"
                state["camera_status"] = "disconnected"
                state["camera_image_status"] = "unknown"
                state["camera_mean_luma"] = None
                state["camera_fps"] = 0.0
                state["pipeline_fps"] = 0.0
                state["infer_ms"] = 0.0
            with latest_frame_lock:
                latest_jpeg = None
                latest_raw_frame = None
            log(f"yolo-worker-error {type(exc).__name__}: {exc}; retrying in 5s")
            broadcast_state()
        finally:
            if global_camera is not None:
                global_camera.stop()
                global_camera = None
        time.sleep(5)


def _camera_pipeline() -> None:
    global latest_jpeg, latest_raw_frame, global_camera
    model = load_yolo_model()
    with state_lock:
        state["yolo_status"] = "waiting_camera"

    camera = None
    last_frame_time = time.monotonic()
    last_count = -1
    metric_frames = 0
    metric_start = time.perf_counter()
    infer_ms_acc = 0.0
    capture_ms_acc = 0.0
    encode_ms_acc = 0.0
    last_seq = -1
    while True:
        if camera is None:
            camera = ThreadedCamera(CAMERA_INDEX)
            if not camera.is_opened():
                camera.stop()
                camera = None
                with state_lock:
                    state["camera_status"] = "disconnected"
                    state["camera_image_status"] = "unknown"
                    state["camera_mean_luma"] = None
                log(f"camera-open-failed index={CAMERA_INDEX}; retrying in 5s")
                broadcast_state()
                time.sleep(5)
                continue
            global_camera = camera
            camera.start()
            last_seq = -1
            last_frame_time = time.monotonic()
            with state_lock:
                state["camera_status"] = "connecting"
            log(camera.describe())
        frame_start = time.perf_counter()
        ok, frame, camera_fps, seq = camera.read()
        after_capture = time.perf_counter()
        if not ok or frame is None or seq == last_seq:
            if time.monotonic() - last_frame_time > 5:
                log("camera-frame-timeout; reconnecting")
                camera.stop()
                camera = None
                global_camera = None
                with latest_frame_lock:
                    latest_jpeg = None
                    latest_raw_frame = None
                with state_lock:
                    state["camera_status"] = "disconnected"
                    state["camera_image_status"] = "unknown"
                    state["camera_mean_luma"] = None
                    state["yolo_status"] = "waiting_camera"
                    state["camera_fps"] = 0.0
                    state["pipeline_fps"] = 0.0
                broadcast_state()
                time.sleep(5)
                continue
            time.sleep(0.02)
            continue
        last_seq = seq
        last_frame_time = time.monotonic()
        height, width = frame.shape[:2]
        with torch.inference_mode():
            results = model(
                frame,
                classes=[0],
                verbose=False,
                device=YOLO_DEVICE,
                imgsz=INFER_IMGSZ,
            )
        after_infer = time.perf_counter()
        person_count = len(results[0].boxes)
        boxes = boxes_from_result(results[0], width, height)
        image_status, mean_luma = camera_image_health(frame)
        with state_lock:
            health_changed = (
                state["camera_image_status"] != image_status
                or state["yolo_status"] != "running"
            )
            state["camera_status"] = "connected"
            state["camera_image_status"] = image_status
            state["camera_mean_luma"] = mean_luma
            state["yolo_status"] = "running"
            state["yolo_error"] = ""
            state["yolo_last_success"] = time.time()
            current_lux = state.get("lux_avg")
        if health_changed:
            log(f"camera-image-status={image_status} mean_luma={mean_luma:.2f}")
            broadcast_state()
        with state_lock:
            candidate_due = (_auto_candidate_level is not None
                             and time.monotonic() - _auto_candidate_since >= AUTO_LEVEL_STABLE_SEC)
        if person_count != last_count or candidate_due:
            set_level_from_count(person_count)
            if person_count != last_count:
                last_count = person_count
                broadcast_state()

        level = choose_auto_level(person_count, current_lux)
        annotated = results[0].plot()
        cv2.putText(
            annotated,
            ("IMAGE TOO DARK | " if image_status == "dark" else "")
            + f"people={person_count} L{level.level} {level.label}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (40, 255, 180),
            2,
            cv2.LINE_AA,
        )

        with latest_frame_lock:
            latest_raw_frame = frame.copy()
            latest_jpeg = encode_jpeg(annotated)
        after_encode = time.perf_counter()

        metric_frames += 1
        capture_ms_acc += after_capture - frame_start
        infer_ms_acc += after_infer - after_capture
        encode_ms_acc += after_encode - after_infer
        metric_elapsed = after_encode - metric_start
        if metric_elapsed >= 5.0:
            pipeline_fps = metric_frames / metric_elapsed
            with state_lock:
                state["camera_fps"] = round(camera_fps, 2)
                state["pipeline_fps"] = round(pipeline_fps, 2)
                state["capture_ms"] = round(capture_ms_acc / metric_frames * 1000, 2)
                state["infer_ms"] = round(infer_ms_acc / metric_frames * 1000, 2)
                state["encode_ms"] = round(encode_ms_acc / metric_frames * 1000, 2)
            log(
                "perf "
                f"camera_fps={camera_fps:.2f} pipeline_fps={pipeline_fps:.2f} "
                f"capture_ms={capture_ms_acc / metric_frames * 1000:.2f} "
                f"infer_ms={infer_ms_acc / metric_frames * 1000:.2f} "
                f"encode_ms={encode_ms_acc / metric_frames * 1000:.2f}"
            )
            metric_frames = 0
            metric_start = time.perf_counter()
            infer_ms_acc = 0.0
            capture_ms_acc = 0.0
            encode_ms_acc = 0.0

        if DetectionSummary is not None:
            summary = DetectionSummary(
                person_count=person_count,
                level=level.level,
                level_label=level.label,
                brightness=level.brightness,
                frame_width=width,
                frame_height=height,
                boxes=boxes,
            )
            maybe_run_vlm(frame, summary)


def mjpeg_stream():
    while True:
        with latest_frame_lock:
            frame = latest_jpeg
        if not frame:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template(
        "index.html", auto_levels=AUTO_LEVELS,
        daylight_lux_threshold=DAYLIGHT_LUX_THRESHOLD,
        mid_people_max=MID_PEOPLE_MAX,
        countdown_seconds=COUNTDOWN_SECONDS, button_pin=PIN_IN,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/status")
def api_status():
    return jsonify(get_public_state())


@app.route("/api/light", methods=["POST"])
def api_light():
    data = request.get_json(force=True, silent=True) or {}
    brightness = int(data.get("brightness", state["brightness"]))
    with state_lock:
        if state["emergency_trigger"]:
            return jsonify({"ok": False, "message": "Emergency is active"}), 409
        state["mode"] = "manual"
        state["brightness"] = max(0, min(100, brightness))
        ok, msg = send_light(state["brightness"], "api-manual-control")
    broadcast_state()
    return jsonify({"ok": ok, "message": msg, "state": get_public_state()})


@app.route("/api/light/<device_id>", methods=["POST"])
def api_light_single(device_id: str):
    """Control a single device by its ID."""
    data = request.get_json(force=True, silent=True) or {}
    brightness = int(data.get("brightness", state["brightness"]))
    
    # Find the target device
    target_device = None
    for device in DEVICES:
        if device["id"] == device_id:
            target_device = device
            break
    
    if target_device is None:
        return jsonify({"ok": False, "message": f"Device {device_id} not found"}), 404
    
    with state_lock:
        if state["emergency_trigger"]:
            return jsonify({"ok": False, "message": "Emergency is active"}), 409
        state["mode"] = "manual"
        state["brightness"] = max(0, min(100, brightness))
        ok, msg = send_light(state["brightness"], "api-single-control", target_device)
    broadcast_state()
    return jsonify({"ok": ok, "message": msg, "state": get_public_state()})


@socketio.on("connect")
def on_connect():
    emit("state", get_public_state())


@socketio.on("set_mode")
def on_set_mode(data):
    mode = data.get("mode", "auto")
    if mode not in ("auto", "manual"):
        return
    with state_lock:
        if state["emergency_trigger"]:
            return
        state["mode"] = mode
        log(f"mode-change mode={mode}")
        if mode == "auto":
            level = choose_auto_level(state["person_count"], state.get("lux_avg"))
            state["level"] = level.level
            state["level_label"] = level.label
            state["brightness"] = level.brightness
            send_light(level.brightness, f"back-to-auto L{level.level} {level.label}", group_id=6)
    broadcast_state()


@socketio.on("set_brightness")
def on_set_brightness(data):
    """Stage brightness in manual mode – does NOT send to light yet."""
    brightness = max(0, min(100, int(data.get("brightness", 0))))
    with state_lock:
        if state["mode"] != "manual" or state["emergency_trigger"]:
            return
        state["pending_brightness"] = brightness
    broadcast_state()


@socketio.on("send_brightness")
def on_send_brightness(data):
    """Send the staged (or explicitly provided) brightness to the light."""
    device_id = data.get("device_id")
    group_id = data.get("group_id")  # Extract group_id from socket data
    
    with state_lock:
        if state["mode"] != "manual" or state["emergency_trigger"]:
            return
        raw = data.get("brightness")
        if raw is not None:
            brightness = max(0, min(100, int(raw)))
        elif state["pending_brightness"] is not None:
            brightness = state["pending_brightness"]
        else:
            brightness = state["brightness"]
        state["brightness"] = brightness
        state["pending_brightness"] = None

        target_device = None
        if device_id:
            for device in DEVICES:
                if device["id"] == device_id:
                    target_device = device
                    break

        # Pass both target_device and group_id to send_light
        send_light(brightness, "web-manual-send", target_device, group_id=group_id)
    broadcast_state()


def _lux_worker() -> None:
    """Read JNC LUX20 continuously via pymodbus ModbusSerialClient."""
    from pymodbus.client import ModbusSerialClient
    
    last_broadcast_avg: Optional[float] = None
    client = None
    missing_port = False
    
    while True:
        if not os.path.exists(USB_LUX_PORT):
            if not missing_port:
                missing_port = True
                if client:
                    client.close()
                    client = None
                _lux_samples.clear()
                with state_lock:
                    state["lux_raw"] = None
                    state["lux_avg"] = None
                    state["lux_error"] = f"USB lux port not found: {USB_LUX_PORT}"
                log(f"lux-port-missing {USB_LUX_PORT}; retrying every 5s")
                broadcast_state()
            time.sleep(5)
            continue
        missing_port = False
        lux = None
        now = time.time()
        try:
            if client is None or not client.connected:
                log(f"lux-modbus-init port={USB_LUX_PORT} baud={USB_LUX_BAUD} slave={USB_LUX_UNIT_ID}")
                client = ModbusSerialClient(
                    port=USB_LUX_PORT,
                    baudrate=USB_LUX_BAUD,
                    parity='N',
                    stopbits=1,
                    bytesize=8,
                    timeout=1
                )
                client.connect()
            
            if client.connected:
                response = client.read_holding_registers(address=0x0000, count=2, slave=USB_LUX_UNIT_ID)
                if response is not None and not response.isError():
                    high_word = response.registers[0]
                    low_word = response.registers[1]
                    lux = float((high_word << 16) | low_word)
                else:
                    err_msg = str(response) if response is not None else "Timeout"
                    # log(f"lux-modbus-read-error: {err_msg}") # Quiet logs to avoid spam
        except Exception as exc:
            log(f"lux-worker-exception: {exc}")
            if client:
                try:
                    client.close()
                except Exception:
                    pass
                client = None
        
        if lux is not None:
            _lux_samples.append((now, lux))
        
        # Drop samples older than the window
        while _lux_samples and now - _lux_samples[0][0] > LUX_WINDOW:
            _lux_samples.popleft()
            
        avg = round(sum(v for _, v in _lux_samples) / len(_lux_samples), 1) if _lux_samples else None
        
        with state_lock:
            state["lux_raw"] = lux
            state["lux_avg"] = avg
            if lux is not None:
                state["lux_last_seen"] = now
                state["lux_failures"] = 0
                state["lux_error"] = ""
            elif state.get("lux_raw") is None:
                state["lux_error"] = "USB lux no response"
                state["lux_failures"] = state.get("lux_failures", 0) + 1
            else:
                state["lux_error"] = "USB lux no response; keeping last value"
                state["lux_failures"] = state.get("lux_failures", 0) + 1
            
        # Only push to clients when value changes meaningfully
        if avg != last_broadcast_avg:
            last_broadcast_avg = avg
            with state_lock:
                pc = state.get("person_count", 0)
            set_level_from_count(pc)
            broadcast_state()
            
        time.sleep(1.0)


def _device_watchdog() -> None:
    """Mark devices as lost if no 0x80 packet received for DEVICE_LOST_TIMEOUT seconds."""
    while True:
        now = time.time()
        changed = False
        lost_names = []
        with state_lock:
            for device in DEVICES:
                dev_id = device["id"]
                last = _device_last_seen.get(dev_id, 0.0)
                # Only flag lost if we've seen the device at least once
                is_lost = last > 0 and (now - last) > DEVICE_LOST_TIMEOUT
                if state["device_lost"].get(dev_id) != is_lost:
                    state["device_lost"][dev_id] = is_lost
                    changed = True
                    if is_lost:
                        lost_names.append(device["name"])
        for name in lost_names:
            log(f"device-lost timeout={DEVICE_LOST_TIMEOUT}s device={name}")
        if changed:
            broadcast_state()
        time.sleep(10)


@socketio.on("set_group")
def on_set_group(data):
    """Assign a device to a group ID."""
    device_id = data.get("device_id")
    group_id = int(data.get("group_id", 0))
    
    with state_lock:
        if state["emergency_trigger"]:
            return
        # Find and update device
        target_device = None
        for d in DEVICES:
            if d["id"] == device_id:
                d["group_id"] = group_id
                target_device = d
                break

        if target_device:
            save_devices(DEVICES)
            packet = build_group_set_packet(device_id, group_id)
            log(f"set-group device={target_device['name']} group={group_id}")
            _start_retry_worker(target_device, packet, "set-group")
    broadcast_state()


def _gpio_worker() -> None:
    """Read the button on BOARD pin 22 and control the relay on BOARD pin 16."""
    if not USE_GPIO:
        log("GPIO monitoring disabled (ENABLE_GPIO=0).")
        return
    if not HAS_GPIO:
        with state_lock:
            state["gpio_status"] = "unavailable"
        log("Jetson.GPIO is not available. GPIO monitoring is disabled.")
        return

    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        # Jetson.GPIO does not configure pull resistors; the input needs a stable bias.
        GPIO.setup(PIN_IN, GPIO.IN)
        
        # Setup each output pin, logging any channel failures without crashing
        active_outs = []
        for pin in PIN_OUTS:
            try:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                active_outs.append(pin)
                log(f"GPIO Pin {pin} output initialized successfully.")
            except Exception as e:
                log(f"GPIO Pin {pin} output setup failed (might be invalid or system-reserved): {e}")
        
        if not active_outs:
            raise RuntimeError("No relay outputs initialized")
        log(f"Jetson.GPIO initialized. Active outputs: {active_outs}")
        with state_lock:
            state["gpio_status"] = "enabled"
    except Exception as e:
        with state_lock:
            state["gpio_status"] = "unavailable"
        log(f"Jetson.GPIO initialization failed: {e}")
        return

    while True:
        try:
            in_state = GPIO.input(PIN_IN)
            # Active-Low: LOW means pressed/triggered
            triggered = (in_state == GPIO.LOW)
            
            # Sync relay outputs with the emergency state
            out_state = GPIO.HIGH if triggered else GPIO.LOW
            for pin in active_outs:
                try:
                    GPIO.output(pin, out_state)
                except Exception as exc:
                    log(f"Failed to output to pin {pin}: {exc}")
            
            with state_lock:
                if state.get("emergency_trigger") != triggered:
                    state["emergency_trigger"] = triggered
                    log(f"Emergency state changed: {triggered} (outputs={out_state})")
                    # Streetlight group linkage
                    if triggered:
                        global _countdown_generation
                        _countdown_generation += 1
                        state["countdown_active"] = False
                        state["pending_level"] = None
                        state["countdown_remaining"] = 0
                        state["level"] = 4
                        state["level_label"] = "emergency"
                        state["brightness"] = 100
                        
                        # Emergency ON → force group 6 to 100% brightness
                        send_light(100, "emergency-on", group_id=6)
                        log("Emergency: streetlight group 6 set to 100%")
                    else:
                        # Emergency OFF → restore auto level brightness
                        with state_lock:
                            pc = state.get("person_count", 0)
                            current_lux = state.get("lux_avg")
                        level = choose_auto_level(pc, current_lux)
                        state["level"] = level.level
                        state["level_label"] = level.label
                        state["brightness"] = level.brightness
                        send_light(level.brightness, f"emergency-off-restore L{level.level}", group_id=6)
                        log(f"Emergency cleared: restoring group 6 to L{level.level} brightness={level.brightness}")
                    broadcast_state()
        except Exception as exc:
            log(f"GPIO-worker-loop-failed: {exc}")
        time.sleep(0.1)


import atexit

global_camera = None

def cleanup():
    if global_camera is not None:
        global_camera.stop()
    if USE_GPIO and HAS_GPIO:
        try:
            GPIO.cleanup()
        except Exception:
            pass

atexit.register(cleanup)

if __name__ == "__main__":
    threading.Thread(target=start_tcp_server, daemon=True).start()
    threading.Thread(target=_central_retry_worker, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=_lux_worker, daemon=True).start()
    threading.Thread(target=_device_watchdog, daemon=True).start()
    threading.Thread(target=_gpio_worker, daemon=True).start()
    try:
        socketio.run(app, host=APP_HOST, port=APP_PORT, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
