# Smart Street Light Demo

Jetson 路燈展示服務。網站由 Flask 和 Socket.IO 提供，YOLO26n 在 Jetson 上偵測行人，照度計透過 USB／Modbus 回報照度。這個部署版本使用 YOLO，VLM 關閉。

## 功能與連接埠

- 網頁與 API 使用 TCP 5050，Gateway 以 TCP Client 連到 Jetson 的 5566。
- 網頁支援手動單燈控制、第 6 組原生群控、群組設定、亮度回報、封包紀錄、即時影像及斷線恢復。
- 自動調光依行人數與照度選擇 L0–L3。GPIO 警報會切到 L4，亮度指令及模式切換在警報解除前暫停。群控部分成功時，程式會單獨補送還未確認的燈。
- GPIO 使用 BOARD pin 22 接收警報按鈕（低電位觸發），BOARD pin 16 驅動高電位有效的繼電器。Pin 14 是接地。按鈕輸入需要外部穩定偏壓。

| 等級 | 條件 | 亮度 |
|---|---|---|
| L0 | 照度高於 270 lux | 0% |
| L1 | 0 人 | 10% |
| L2 | 1–3 人 | 30% |
| L3 | 4 人以上 | 60% |
| L4 | 實體警報按下 | 100% |

照度未取得時依人數判斷。一般自動降級需持續 15 秒，升級需穩定 1 秒。

## Docker（Jetson）

映像適用於 ARM64 Jetson、JetPack 5.1.2／L4T R35.4.1，使用 NVIDIA Container Runtime。使用 NVIDIA 的 CUDA PyTorch 與 Ultralytics YOLO26n；建置時下載並驗證與既有 Jetson 部署相同的 YOLO 權重。映像只封裝路燈服務與 YOLO 推論所需元件，VLM 預設關閉。

### 建置

~~~bash
docker build -t smart-streetlight-demo:jetpack5-novlm .
~~~

若 Docker 26 安裝仍使用 legacy builder，可使用 `DOCKER_BUILDKIT=0 docker build -t smart-streetlight-demo:jetpack5-novlm .`。

### 隔離測試

在同一台已由 PM2 提供服務的 Jetson 上，用 15050 驗證容器首頁與狀態 API：

~~~bash
docker run --rm --runtime nvidia \
  -p 127.0.0.1:15050:5050 \
  smart-streetlight-demo:jetpack5-novlm
~~~

此模式適合驗證 HTTP、Socket.IO 與 CUDA 環境。容器沒掛攝影機和照度計時，狀態頁會回報裝置尚未連線。

### 實機部署

正式使用容器前，先在維護時段釋放原本使用 5050／5566、攝影機、照度計與 GPIO 的程序。使用 host network，讓 Gateway 連到原本的 Jetson 位址，路燈程式也能讀到 Gateway 來源 IP。

~~~bash
cp --no-clobber devices.json devices.local.json

docker run -d --name smart-streetlight-demo --restart unless-stopped \
  --runtime nvidia --network host \
  --device /dev/video0 --device /dev/ttyUSB0 \
  --device /dev/gpiochip0 \
  --mount type=bind,source="$(pwd)/devices.local.json",target=/opt/streetlight/devices.json \
  -e ENABLE_GPIO=1 -e JETSON_MODEL_NAME=JETSON_ORIN_NX \
  -e YOLO_DEVICE=cuda:0 \
  smart-streetlight-demo:jetpack5-novlm
~~~

`JETSON_MODEL_NAME` 和裝置路徑對應目前使用的 Orin NX。其他 Jetson 請依型號與實際 GPIO 裝置調整。綁定本機 `devices.local.json` 能讓群組設定跨容器重建保存。Docker 預設以 `/dev/ttyUSB0`、9600 baud 和 Modbus 位址 77 讀取照度計。

## 原生 PM2 服務

已配置 NVIDIA PyTorch、Jetson.GPIO 和 Python 依賴的 Jetson 可繼續使用 PM2。把 `yolo26n.pt` 放在 `models/`，現有 `ecosystem.config.js` 會使用該路徑。

~~~bash
pm2 start ecosystem.config.js
pm2 logs streetlight-lite-agx
~~~

## 迴歸測試

~~~bash
python3 -m unittest test_brightness test_level_guide test_gpio test_yolo_pipeline
node test_brightness_ui.js
~~~
