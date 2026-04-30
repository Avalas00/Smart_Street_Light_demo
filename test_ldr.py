import serial
import time
import sys
import re

# 嘗試常見的連接埠與波特率
PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
BAUDS = [9600, 115200, 57600, 19200]

def test_serial():
    print("開始掃描 USB Serial 裝置...")
    
    for port in PORTS:
        for baud in BAUDS:
            try:
                # 嘗試開啟連接埠
                print(f"嘗試連線: {port} (Baud: {baud})...")
                ser = serial.Serial(port, baud, timeout=2)
                time.sleep(2) # 等待 Arduino 重新啟動（若有）
                
                # 讀取幾行資料看看
                lines_read = 0
                max_attempts = 10
                attempts = 0
                while lines_read < 3 and attempts < max_attempts:
                    if ser.in_waiting > 0:
                        raw_data = ser.readline()
                        try:
                            text = raw_data.decode('utf-8', errors='ignore').strip()
                            if text:
                                # 嘗試解析數值
                                match = re.search(r"[-+]?\d*\.\d+|\d+", text)
                                parsed_val = match.group() if match else "無法解析數字"
                                
                                print(f"✅ 成功收到資料 [{port} @ {baud}]: 原始='{text}' | 解析出數值='{parsed_val}'")
                                lines_read += 1
                        except Exception as e:
                            print(f"⚠️ 收到無法解析的資料: {raw_data}")
                            break
                    else:
                        time.sleep(0.5)
                        attempts += 1
                
                ser.close()
                if lines_read > 0:
                    print(f"\n🎉 找到正確的設定: Port = {port}, Baud = {baud}")
                    return True
                else:
                    print(f"⚠️ {port} @ {baud} 沒有傳送任何資料。")
                    
            except serial.SerialException as e:
                if "Permission denied" in str(e):
                    print(f"❌ 權限不足無法存取 {port}。請嘗試使用 'sudo python3 test_ldr.py' 執行此腳本。")
                    break # 同一個 port 權限不足，換 baud rate 也沒用，跳到下一個 port
                else:
                    # 找不到裝置等錯誤
                    pass
            except Exception as e:
                pass
                
    print("\n❌ 掃描結束。無法從任何常見的 USB 連接埠讀取到資料。")
    print("請確認：\n1. USB 是否確實插妥。\n2. 感測器是否有輸出資料。\n3. 若有權限問題，請加上 sudo 執行。")
    return False

if __name__ == '__main__':
    test_serial()
