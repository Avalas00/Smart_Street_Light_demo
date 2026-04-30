# AGX Streetlight Lite

This is a trimmed streetlight edge app for the AGX.

## What app.py does

- Runs a Flask + Socket.IO web UI on `APP_PORT` (default `5050`).
- Runs a TCP server on `TCP_PORT` (default `5566`) for the streetlight device.
- Reads `/dev/video0` by default and runs YOLO26n person detection.
- Uses the bundled model at `models/yolo26n.pt`.
- Runs YOLO on `cuda:0` through the project `.venv` when CUDA is available.
- Shows the live YOLO stream on the web page.
- Computes streetlight level from person count:
  - L0 standby: 0 people, 20%
  - L1 person-near: 1 person, 55%
  - L2 people-flow: 2-3 people, 75%
  - L3 high-alert: 4+ people, 100%
- Sends dimming packets to `LIGHT_TARGET_IP` (default `192.168.50.202`).
- Supports auto/manual mode from the web page.
- Supports manual dimming from the web page slider.
- Publishes status and packet logs to the web page.
- Calls the VLM harness only when `ENABLE_VLM=1`.

## Streetlight packet

The packet format follows `/home/lucas/Desktop/LungPin/app.py`:

```text
02 0B 52 00 24 E9 79 D6 01 <brightness 0-100> <checksum>
```

`checksum = sum(first 10 bytes) % 256`.

## What Gemma does

`gemma4:e2b` is used as an image-language safety judge through Ollama. It receives:

- One JPEG frame.
- YOLO metadata from the harness: person count, level, brightness, frame size, boxes, trigger reason.

It returns JSON:

```json
{
  "alert": true,
  "risk_level": "medium",
  "abnormal_type": "fallen_pose",
  "message": "疑似有人跌倒，請確認現場。",
  "recommended_brightness": 100,
  "confidence": 0.82
}
```

Gemma is not in the direct dimming control path. YOLO still controls auto brightness. Gemma adds semantic abnormal-behavior warning.

## VLM harness design

The harness is in `vlm_harness.py`, and the model adapter is in `vlm_alert.py`.

The app calls only:

```python
trigger = vlm_harness.begin(summary)
alert = vlm_harness.inspect(image_jpeg, summary, trigger)
vlm_harness.finish()
```

This makes VLM easy to remove:

```bash
ENABLE_VLM=0 python3 app.py
```

or remove `vlm_harness.py` / `vlm_alert.py`; the YOLO and streetlight path stays independent.

Current harness triggers:

- person present (`VLM_TRIGGER_PEOPLE`, default `1`)
- YOLO level threshold (`VLM_TRIGGER_LEVEL`, default `1`)
- crowd check (`person_count >= 3`)
- fallen pose candidate from wide person box aspect ratio
- cooldown via `VLM_MIN_INTERVAL_SEC` (default `20`)

## PM2

Start:

```bash
cd /home/lucas/Desktop/streetlight_lite_agx
pm2 start ecosystem.config.js
pm2 save
```

Logs:

```bash
pm2 logs streetlight-lite-agx
```

Restart:

```bash
pm2 restart streetlight-lite-agx
```

Stop:

```bash
pm2 stop streetlight-lite-agx
```

## Run without PM2

```bash
cd /home/lucas/Desktop/streetlight_lite_agx
APP_PORT=5050 TCP_PORT=5566 ENABLE_VLM=1 VLM_MODEL=gemma4:e2b python3 app.py
```

AGX currently has Ollama `0.21.2` and `gemma4:e2b` downloaded.
# smart_street_light_demo
