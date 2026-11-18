module.exports = {
  apps: [
    {
      name: "streetlight-lite-agx",
      cwd: __dirname,
      script: "app.py",
      interpreter: "/usr/bin/python3",
      autorestart: true,
      min_uptime: "10s",
      restart_delay: 5000,
      watch: false,
      max_restarts: 10,
      env: {
        APP_HOST: "0.0.0.0",
        APP_PORT: "5050",
        TCP_HOST: "0.0.0.0",
        TCP_PORT: "5566",
        LIGHT_TARGET_IP: "192.168.50.202",
        CAMERA_INDEX: "0",
        ENABLE_GPIO: "1",
        CAMERA_FOURCC: "MJPG",
        CAMERA_FPS: "30",
        FRAME_WIDTH: "960",
        FRAME_HEIGHT: "540",
        INFER_IMGSZ: "640",
        YOLO_MODEL: require("path").join(__dirname, "models", "yolo26n.pt"),
        YOLO_DEVICE: "cuda:0",
        LD_LIBRARY_PATH: "/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu",
        USB_LUX_PORT: "/dev/ttyUSB0",
        USB_LUX_BAUD: "9600",
        USB_LUX_UNIT_ID: "77",
        ENABLE_VLM: "0"
      }
    }
  ]
};
