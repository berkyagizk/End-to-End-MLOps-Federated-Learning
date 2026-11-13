import subprocess
import time
import argparse
import sys

import pandas as pd

from src.utils import build_partition_plan, validate_client_count

# Terminalden banka sayısını al
parser = argparse.ArgumentParser(description="Federe Öğrenme Simülasyon Otomasyonu")
parser.add_argument("--num-clients", type=int, required=True, help="Simüle edilecek banka (client) sayısı")
parser.add_argument("--seed", type=int, default=42, help="Veri bölme tohumu")
args = parser.parse_args()
try:
    validate_client_count(args.num_clients)
    # Reject infeasible splits before launching a server that would wait for clients.
    build_partition_plan(pd.read_csv("data/creditcard.csv"), args.num_clients, seed=args.seed)
except (OSError, ValueError) as error:
    parser.error(str(error))

print(f"\nSİMÜLASYON BAŞLIYOR: 1 Sunucu ve {args.num_clients} Banka\n")

# 1. Sunucuyu (Server) Başlat
print("-> Sunucu (Server) ayağa kaldırılıyor...")
server_process = subprocess.Popen([sys.executable, "src/server.py", "--num-clients", str(args.num_clients)])
time.sleep(4) # Sunucunun portu dinlemeye başlaması için 4 saniye bekle

# 2. İstemcileri (Client) Başlat
client_processes = []
for i in range(args.num_clients):
    print(f"-> Banka Şubesi {i} başlatılıyor...")
    cmd = [sys.executable, "src/client.py", "--client-id", str(i),
           "--num-clients", str(args.num_clients), "--seed", str(args.seed)]
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
