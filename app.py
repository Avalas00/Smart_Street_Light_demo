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
from flask_socketio import SocketIO
from ultralytics import YOLO

try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    GPIO = None

PIN_IN = 22
PIN_OUTS = [14, 16]

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
    "/home/lucas/Desktop/streetlight_lite_agx/models/yolo26n.pt",
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
    LightLevel(1, "low_pedestrian", 15),
    LightLevel(2, "mid_pedestrian", 35),
    LightLevel(3, "high_pedestrian", 60),
]
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
    "device_brightness": {d["id"]: AUTO_LEVELS[0].brightness for d in DEVICES},
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
    "device_target_brightness": {d["id"]: AUTO_LEVELS[0].brightness for d in DEVICES},
    "device_last_command_time": {d["id"]: 0.0 for d in DEVICES},
    "camera_fps": 0.0,
    "pipeline_fps": 0.0,
    "infer_ms": 0.0,
    "capture_ms": 0.0,
    "encode_ms": 0.0,
    "camera_fourcc": CAMERA_FOURCC,
    "camera_requested_fps": CAMERA_FPS,
    "emergency_trigger": False,
    "telemetry_history": [],
}

# Countdown timer control
_countdown_cancel = threading.Event()
_countdown_generation = 0
_retry_lock = threading.Lock()
_retry_inflight: Dict[str, dict] = {}
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
# Per-device last control acknowledgment timestamp
_last_ack_time: Dict[str, float] = {d["id"]: 0.0 for d in DEVICES}
DEVICE_LOST_TIMEOUT = 1000  # 3 minutes

# USB RS-485 Light Sensor (JNC LUX20)
USB_LUX_PORT = os.getenv("USB_LUX_PORT", "/dev/ttyUSB0")
USB_LUX_BAUD = int(os.getenv("USB_LUX_BAUD", "9600"))
USB_LUX_UNIT_ID = int(os.getenv("USB_LUX_UNIT_ID", "77"))
LUX_WINDOW = 5  # rolling-average window in seconds
_lux_samples: deque = deque()  # (timestamp, lux)
_last_lux_error_log = 0.0


def _drop_tcp_client(client_ip: str, reason: str) -> None:
    conn = active_tcp_clients.pop(client_ip, None)
    if conn is not None:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        log(f"device-connection-reset ip={client_ip} reason={reason}")
    device_ips = [d["ip"] for d in DEVICES]
    with state_lock:
        state["device_connected"] = any(ip in active_tcp_clients for ip in device_ips)
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
    # 1. Day mode: if lux > 150, stay at L0 (standby, 0% brightness)
    if lux is not None and lux > 270:
        return AUTO_LEVELS[0]
    
    # 2. Night mode: lux <= 150 (or sensor unavailable)
    if person_count <= 0:
        return AUTO_LEVELS[1] # L1 (low)
    elif person_count <= 2:
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
    devices_to_control = []
    
    if group_id is not None:
        # Group control: Find at least one device that belongs to this group
        representative_device = None
        for d in DEVICES:
            if d.get("group_id") == group_id:
                representative_device = d
                break
        
        if representative_device:
            # Update target tracking for all devices in this group
            now = time.time()
            with state_lock:
                for d in DEVICES:
                    if d.get("group_id") == group_id:
                        state["device_target_brightness"][d["id"]] = brightness
                        state["device_last_command_time"][d["id"]] = now
            
            packet = build_group_control_packet(representative_device["id"], group_id, brightness)
            msg_reason = f"{reason}-group-{group_id}"
            _start_retry_worker(representative_device, packet, msg_reason)
            return True, f"Group {group_id} control started via {representative_device['name']}"
        else:
            return False, f"No devices found in group {group_id}"

    # Individual control
    if target_device:
        devices_to_control = [target_device]
    else:
        devices_to_control = DEVICES

    for device in devices_to_control:
        if not device.get("enabled", True):
            continue
        
        # Update target tracking
        with state_lock:
            state["device_target_brightness"][device["id"]] = brightness
            state["device_last_command_time"][device["id"]] = time.time()
            
        packet = build_light_packet(device["id"], brightness)
        _start_retry_worker(device, packet, reason)

    return True, "Control sequence started"


def _start_retry_worker(device: dict, packet: bytes, reason: str) -> None:
    dev_id = device["id"]
    target_ip = device["ip"]
    packet_hex = packet.hex().upper()
    if target_ip not in active_tcp_clients:
        log(f"control-not-connected: {device['name']} target={target_ip} hex={packet_hex}")
        return

    # Log 0x52 / 0x36 sent packets to history
    # The brightness is at packet[9] for 0x52 (length 11), packet[10] for 0x36 (length 12)
    dimming_val = packet[-2] if len(packet) > 2 else 0
    _append_history(device["name"], packet_hex, f"0x{packet[2]:02X}", dimming_val, time.time())

    with _retry_lock:
        current = _retry_inflight.get(dev_id)
        if current and current.get("packet_hex") == packet_hex and current.get("active"):
            log(f"retry-skip-duplicate: {device['name']} hex={packet_hex}")
            return
        token = time.time()
        _retry_inflight[dev_id] = {
            "device": device,
            "packet": packet,
            "packet_hex": packet_hex,
            "reason": reason,
            "token": token,
            "active": True,
            "phase": 1,
            "attempts": 0,
            "next_send_time": 0.0,
        }


def get_public_state() -> dict:
    with state_lock:
        public = dict(state)
    public["clients"] = sorted(active_tcp_clients.keys())
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
        # Emergency override: skip all auto-level logic while emergency is active
        if state.get("emergency_trigger"):
            state["person_count"] = person_count
            return
        current_lux = state.get("lux_avg")
        current_level_idx = state["level"]
        is_auto = state["mode"] == "auto"

    target_level = choose_auto_level(person_count, current_lux)
    now = time.time()
    
    with state_lock:
        state["person_count"] = person_count
        if not is_auto:
            if state["countdown_active"] or state["pending_level"] is not None:
                _countdown_generation += 1
                state["countdown_active"] = False
                state["pending_level"] = None
                state["countdown_remaining"] = 0
                broadcast_state()
            return

        # 1. UPGRADE Logic (target > current) -> Immediate (with candidate stability)
        if target_level.level > current_level_idx:
            # Require stable candidate for upgrades to avoid flickering
            if _auto_candidate_level != target_level.level:
                _auto_candidate_level = target_level.level
                _auto_candidate_since = now
                return
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
                log(f"countdown-start: waiting 30s to downgrade to L{target_level.level}")
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
                active_tcp_clients[client_ip] = conn
                # Check if any device IP is connected
                device_ips = [d["ip"] for d in DEVICES]
                with state_lock:
                    state["device_connected"] = any(ip in active_tcp_clients for ip in device_ips)
                log(f"device-connected {client_ip}:{addr[1]}")
                broadcast_state()
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
        # Add to telemetry history
        _append_history(matched_device["name"], packet_hex, f"0x{cmd:02X}", brightness, now,
                        voltage, current, active_p, pf, rssi)

        with state_lock:
            state["device_brightness"][dev_id] = brightness
            state["device_detail"][dev_id] = detail
            
            target = state["device_target_brightness"].get(dev_id)
            last_cmd_t = state["device_last_command_time"].get(dev_id, 0.0)

        log(f"device-status ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} brightness={brightness}")
        socketio.emit("device_brightness_update", {"device_id": dev_id, "brightness": brightness, "cmd": f"0x{cmd:02X}"})
        socketio.emit("device_detail_update", detail)

        if target is not None and brightness == target:
            _last_ack_time[dev_id] = now

        # Inconsistency check
        if target is not None and brightness != target:
            if now - last_cmd_t > 10.0:
                log(f"consistency-sync: {matched_device['name']} reported={brightness} target={target}. Re-sending...")
                send_light(target, "consistency-sync", target_device=matched_device)

    elif cmd == 0x40 and len(packet) >= 14:
        brightness = packet[13]
        _append_history(matched_device["name"], packet_hex, f"0x{cmd:02X}", brightness, now)
        with state_lock:
            state["device_brightness"][dev_id] = brightness
            target = state["device_target_brightness"].get(dev_id)
            last_cmd_t = state["device_last_command_time"].get(dev_id, 0.0)
            # Sync dimming into device_detail and mark telemetry as stale
            # (power/voltage values are from the previous 0x80 packet)
            if dev_id in state["device_detail"] and state["device_detail"][dev_id]:
                state["device_detail"][dev_id]["dimming"] = brightness
                state["device_detail"][dev_id]["telemetry_stale"] = True

        log(f"device-status ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} brightness={brightness}")
        socketio.emit("device_brightness_update", {"device_id": dev_id, "brightness": brightness, "cmd": f"0x{cmd:02X}"})
        if target is not None and brightness == target:
            _last_ack_time[dev_id] = now
        if target is not None and brightness != target:
            if now - last_cmd_t > 10.0:
                log(f"consistency-sync: {matched_device['name']} reported={brightness} target={target}. Re-sending...")
                send_light(target, "consistency-sync", target_device=matched_device)
    elif cmd in (0x41, 0x53, 0x37):
        _last_ack_time[dev_id] = now

        if len(packet) >= 14:
            # For 0x41, 0x53, 0x37, brightness is at index 13
            brightness = packet[13]
            _append_history(matched_device["name"], packet_hex, f"0x{cmd:02X}", brightness, now)
            with state_lock:
                state["device_brightness"][dev_id] = brightness
                target = state["device_target_brightness"].get(dev_id)
                last_cmd_t = state["device_last_command_time"].get(dev_id, 0.0)
                if dev_id in state["device_detail"] and state["device_detail"][dev_id]:
                    state["device_detail"][dev_id]["dimming"] = brightness
                    state["device_detail"][dev_id]["telemetry_stale"] = True

            log(f"device-ack-update ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} brightness={brightness}%")
            socketio.emit("device_brightness_update", {"device_id": dev_id, "brightness": brightness, "cmd": f"0x{cmd:02X}"})

            if target is not None and brightness == target:
                _last_ack_time[dev_id] = now
            if target is not None and brightness != target:
                if now - last_cmd_t > 10.0:
                    log(f"consistency-sync: {matched_device['name']} reported={brightness} target={target}. Re-sending...")
                    g_id = matched_device.get("group_id")
                    if g_id is not None:
                        send_light(target, "consistency-sync", group_id=g_id)
                    else:
                        send_light(target, "consistency-sync", target_device=matched_device)
        else:
            log(f"device-ack ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']}")
    else:
        log(f"device-rx-unhandled ip={client_ip} cmd=0x{cmd:02X} device={matched_device['name']} hex={packet_hex}")
        
    broadcast_state()


def _central_retry_worker() -> None:
    """Central polling worker for sending retries continuously without blocking."""
    while True:
        now = time.time()
        to_send = []
        with _retry_lock:
            for dev_id, item in list(_retry_inflight.items()):
                if not item.get("active"):
                    continue
                
                device = item["device"]
                
                # Check if acked
                if _last_ack_time.get(dev_id, 0) >= item["token"]:
                    log(f"retry-success: {device['name']} acknowledged in p{item['phase']}")
                    _retry_inflight.pop(dev_id, None)
                    continue

                if item["phase"] == 1:
                    if item["attempts"] >= 3:
                        log(f"retry-p2-wait: phase 1 failed, waiting 5s for {device['name']}")
                        item["phase"] = 2
                        item["next_send_time"] = now + 5.0
                        item["attempts"] = 0
                        continue
                    if now >= item["next_send_time"]:
                        to_send.append(item)
                        item["attempts"] += 1
                        item["next_send_time"] = now + 2.0
                        
                elif item["phase"] == 2:
                    if item["attempts"] >= 3:
                        log(f"retry-exhausted: {device['name']} failed to acknowledge after all attempts")
                        _retry_inflight.pop(dev_id, None)
                        continue
                    if now >= item["next_send_time"]:
                        to_send.append(item)
                        item["attempts"] += 1
                        item["next_send_time"] = now + 5.0

        for item in to_send:
            device = item["device"]
            packet = item["packet"]
            hex_str = packet.hex().upper()
            target_ip = device["ip"]
            log(f"retry-p{item['phase']}: {device['name']} hex={hex_str}")
            
            conn = active_tcp_clients.get(target_ip)
            if conn:
                try:
                    conn.sendall(packet)
                except Exception:
                    pass
            time.sleep(0.05)

        time.sleep(0.1)


def handle_tcp_client(conn: socket.socket, client_ip: str) -> None:
    buffer = b""
    try:
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data
                while len(buffer) >= 2:
                    if buffer[0] != 0x02:
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
        active_tcp_clients.pop(client_ip, None)
        # Check if any device IP is still connected
        device_ips = [d["ip"] for d in DEVICES]
        with state_lock:
            state["device_connected"] = any(ip in active_tcp_clients for ip in device_ips)
        log(f"device-disconnected {client_ip}")
        broadcast_state()


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
        self.thread = threading.Thread(target=self.update, daemon=True)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def start(self) -> "ThreadedCamera":
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
        while True:
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


def camera_worker() -> None:
    global latest_jpeg, latest_raw_frame
    log(
        "loading-yolo "
        f"model={YOLO_MODEL} device={YOLO_DEVICE} "
        f"torch={torch.__version__} cuda={torch.cuda.is_available()}"
    )
    model = YOLO(YOLO_MODEL)
    if hasattr(model, "fuse"):
        try:
            model.fuse()
        except Exception:
            pass

    camera = ThreadedCamera(CAMERA_INDEX)
    if not camera.is_opened():
        log(f"camera-open-failed index={CAMERA_INDEX}")
        return
    camera.start()
    log(camera.describe())

    last_count = -1
    metric_frames = 0
    metric_start = time.perf_counter()
    infer_ms_acc = 0.0
    capture_ms_acc = 0.0
    encode_ms_acc = 0.0
    last_seq = -1
    while True:
        frame_start = time.perf_counter()
        ok, frame, camera_fps, seq = camera.read()
        after_capture = time.perf_counter()
        if not ok or frame is None:
            time.sleep(0.02)
            continue
        if seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = seq

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
        with state_lock:
            current_lux = state.get("lux_avg")
        level = choose_auto_level(person_count, current_lux)
        annotated = results[0].plot()
        cv2.putText(
            annotated,
            f"people={person_count} L{level.level} {level.label}",
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

        if person_count != last_count:
            last_count = person_count
            set_level_from_count(person_count)

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
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(get_public_state())


@app.route("/api/light", methods=["POST"])
def api_light():
    data = request.get_json(force=True, silent=True) or {}
    brightness = int(data.get("brightness", state["brightness"]))
    with state_lock:
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
        state["mode"] = "manual"
        state["brightness"] = max(0, min(100, brightness))
    
    ok, msg = send_light(state["brightness"], "api-single-control", target_device)
    broadcast_state()
    return jsonify({"ok": ok, "message": msg, "state": get_public_state()})


@socketio.on("connect")
def on_connect():
    socketio.emit("state", get_public_state())


@socketio.on("set_mode")
def on_set_mode(data):
    mode = data.get("mode", "auto")
    if mode not in ("auto", "manual"):
        return
    with state_lock:
        state["mode"] = mode
        person_count = state["person_count"]
    log(f"mode-change mode={mode}")
    if mode == "auto":
        with state_lock:
            current_lux = state.get("lux_avg")
        level = choose_auto_level(person_count, current_lux)
        with state_lock:
            state["level"] = level.level
            state["level_label"] = level.label
            state["brightness"] = level.brightness
        # Use Group 6 when switching back to auto
        send_light(level.brightness, f"back-to-auto L{level.level} {level.label}", group_id=6)
    broadcast_state()


@socketio.on("set_brightness")
def on_set_brightness(data):
    """Stage brightness in manual mode – does NOT send to light yet."""
    brightness = max(0, min(100, int(data.get("brightness", 0))))
    with state_lock:
        if state["mode"] != "manual":
            return
        state["pending_brightness"] = brightness
    broadcast_state()


@socketio.on("send_brightness")
def on_send_brightness(data):
    """Send the staged (or explicitly provided) brightness to the light."""
    device_id = data.get("device_id")
    group_id = data.get("group_id")  # Extract group_id from socket data
    
    with state_lock:
        if state["mode"] != "manual":
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
    
    while True:
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
    """Monitor GPIO PIN_IN (22) and control PIN_OUTS (14, 16)."""
    if not HAS_GPIO:
        log("Jetson.GPIO is not available. GPIO monitoring is disabled.")
        return

    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(PIN_IN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Setup each output pin, logging any channel failures without crashing
        active_outs = []
        for pin in PIN_OUTS:
            try:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                active_outs.append(pin)
                log(f"GPIO Pin {pin} output initialized successfully.")
            except Exception as e:
                log(f"GPIO Pin {pin} output setup failed (might be invalid or system-reserved): {e}")
        
        log(f"Jetson.GPIO initialized. Active outputs: {active_outs}")
    except Exception as e:
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
                    broadcast_state()
                    # Streetlight group linkage
                    if triggered:
                        # Emergency ON → force group 6 to 100% brightness
                        send_light(100, "emergency-on", group_id=6)
                        log("Emergency: streetlight group 6 set to 100%")
                    else:
                        # Emergency OFF → restore auto level brightness
                        with state_lock:
                            pc = state.get("person_count", 0)
                            current_lux = state.get("lux_avg")
                        level = choose_auto_level(pc, current_lux)
                        send_light(level.brightness, f"emergency-off-restore L{level.level}", group_id=6)
                        log(f"Emergency cleared: restoring group 6 to L{level.level} brightness={level.brightness}")
        except Exception as exc:
            log(f"GPIO-worker-loop-failed: {exc}")
        time.sleep(0.1)


if __name__ == "__main__":
    threading.Thread(target=start_tcp_server, daemon=True).start()
    threading.Thread(target=_central_retry_worker, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=_lux_worker, daemon=True).start()
    threading.Thread(target=_device_watchdog, daemon=True).start()
    threading.Thread(target=_gpio_worker, daemon=True).start()
    socketio.run(app, host=APP_HOST, port=APP_PORT, allow_unsafe_werkzeug=True)
