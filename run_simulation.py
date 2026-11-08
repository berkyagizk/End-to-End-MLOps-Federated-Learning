import subprocess
import time
import argparse
import sys

# Terminalden banka sayısını al
parser = argparse.ArgumentParser(description="Federe Öğrenme Simülasyon Otomasyonu")
parser.add_argument("--num-clients", type=int, required=True, help="Simüle edilecek banka (client) sayısı")
args = parser.parse_args()

print(f"\nSİMÜLASYON BAŞLIYOR: 1 Sunucu ve {args.num_clients} Banka\n")

# 1. Sunucuyu (Server) Başlat
print("-> Sunucu (Server) ayağa kaldırılıyor...")
server_process = subprocess.Popen([sys.executable, "src/server.py", "--num-clients", str(args.num_clients)])
time.sleep(4) # Sunucunun portu dinlemeye başlaması için 4 saniye bekle

# 2. İstemcileri (Client) Başlat
client_processes = []
for i in range(args.num_clients):
    print(f"-> Banka Şubesi {i} başlatılıyor...")
    cmd = [sys.executable, "src/client.py", "--client-id", str(i), "--num-clients", str(args.num_clients)]
    process = subprocess.Popen(cmd)
    client_processes.append(process)
    time.sleep(1) # Aynı anda yüklenip çakışmamaları için ufak bir bekleme

# 3. İstemcilerin eğitimleri bitirmesini bekle
print("\n Sistem şu an eğitimde... Lütfen bitene kadar bekleyin.\n")
for process in client_processes:
    process.wait()

# 4. Eğitim bitince sunucuyu kapat
server_process.terminate()
print("\n Simülasyon Başarıyla Tamamlandı! Sonuçlar MLflow'a gönderildi.")