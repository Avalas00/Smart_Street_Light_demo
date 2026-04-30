import minimalmodbus
import serial
import time
import sys

# 常見設定
PORTS = ['/dev/ttyUSB0'] # 鎖定這個 port，因為剛才已經確認這條線接在這
BAUDS = [9600, 4800, 19200, 115200, 57600] # 加入 4800，部分 RS485 感測器預設為 4800
SLAVE_IDS = [1, 2, 3] # 預設通常是 1

def scan_modbus():
    print("開始掃描 RS485 Modbus 感測器 (尋找 Lux 數值)...")
    
    for port in PORTS:
        for baud in BAUDS:
            for slave_id in SLAVE_IDS:
                try:
                    # 設定 Modbus RTU 儀器
                    instrument = minimalmodbus.Instrument(port, slave_id)
                    instrument.serial.baudrate = baud
                    instrument.serial.bytesize = 8
                    instrument.serial.parity = serial.PARITY_NONE
                    instrument.serial.stopbits = 1
                    instrument.serial.timeout = 0.5 # 縮短 timeout 加快掃描
                    
                    # 依據手冊，Lux 值在暫存器 0x0000 (High Word) 和 0x0001 (Low Word)
                    # 總共 32-bit (2個暫存器)，使用 function code 03
                    print(f"嘗試連線: Baud={baud}, Slave ID={slave_id} ...", end="", flush=True)
                    
                    # 讀取 32 位元無號整數 (2 個暫存器)
                    # read_long 預設使用 function code 03
                    lux_value = instrument.read_long(registeraddress=0x0000, functioncode=3, signed=False)
                    
                    print(f" 成功！")
                    print(f"\n🎉 成功讀取資料！")
                    print(f"設定: Port={port}, Baud={baud}, Slave_ID={slave_id}")
                    print(f"✅ 當前照度 (Lux): {lux_value}")
                    
                    return True
                    
                except minimalmodbus.NoResponseError:
                    print(" 無回應 (Timeout)")
                    pass
                except minimalmodbus.InvalidResponseError:
                    print(" 回應格式錯誤 (可能是 Baud rate 錯誤)")
                    pass
                except serial.SerialException as e:
                    if "Permission denied" in str(e):
                        print(f"\n❌ 權限不足無法存取 {port}。請加上 sudo 執行。")
                        return False # 權限不足直接中斷
                except Exception as e:
                    print(f" 發生錯誤: {e}")
                    
    print("\n❌ 掃描結束。無法找到 Modbus 裝置！")
    print("【排解步驟】")
    print("1. RS485 常常會接反：請把 A 腳位和 B 腳位對調再試一次（接反不會燒壞，只是沒訊號）。")
    print("2. 檢查供電：感測器是否有正確接上 12V 或 24V 電源？（大多數 RS485 感測器需要外部供電）。")
    print("3. 確認 USB 轉 RS485 的接線是否有鬆動。")
    return False

if __name__ == '__main__':
    scan_modbus()
