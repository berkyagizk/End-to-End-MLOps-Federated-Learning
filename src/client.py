import flwr as fl
import numpy as np
import mlflow
import joblib
import argparse  # Terminalden ID ve Client sayısını almak için eklendi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from utils import load_and_partition_data
from imblearn.over_sampling import SMOTE

# 1. Argüman okuyucu kurulumu (Terminalden --client-id ve --num-clients alacak)
parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument("--client-id", type=int, required=True, help="Bankanın ID numarası (0, 1, 2...)")
parser.add_argument("--num-clients", type=int, required=True, help="Toplam banka sayısı")
args = parser.parse_args()

# 2. Dinamik veri yükleme (Artık her banka KENDİ verisini alacak)
data = load_and_partition_data("data/creditcard.csv", num_clients=args.num_clients)
(X_train, y_train, X_test, y_test) = data[args.client_id] 

model = LogisticRegression(max_iter=1)

try:
    idx_0 = np.where(y_train == 0)[0][0]
    idx_1 = np.where(y_train == 1)[0][0]
    model.fit(X_train[[idx_0, idx_1]], y_train[[idx_0, idx_1]])
    print(f"Client {args.client_id}: Model başarıyla iki sınıfla ilklendirildi.")
except IndexError:
    print(f"Client {args.client_id} HATASI: Bu bankanın verisinde dolandırıcılık örneği (1) bulunamadı!")
    print("Lütfen utils.py dosyasındaki Stratified split kısmını kontrol et.")

class FraudClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        if hasattr(model, "coef_"):
            return [model.coef_, model.intercept_]
        else:
            return []

    def fit(self, parameters, config):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        
        # --- SMOTE ENTEGRASYONU ---
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        model.fit(X_train_balanced, y_train_balanced)
        print(f"Client {args.client_id}: Eğitim tamamlandı (SMOTE ile dengelendi)...")
        
        # Sunucuya yeni (dengeli) veri uzunluğunu yolluyoruz, böylece sunucu doğru ağırlıklı ortalama yapabiliriz.
        return [model.coef_, model.intercept_], len(X_train_balanced), {}

    def evaluate(self, parameters, config):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        print(f"Client {args.client_id} - Test Başarımı: {accuracy}")
        
        # ÇAKIŞMA ÇÖZÜMÜ: Sadece 0. Client modeli kaydetsin
        if args.client_id == 0:
            joblib.dump(model, "best_model.pkl")
            print("Model başarıyla 'best_model.pkl' olarak kaydedildi.")

        return loss, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FraudClient())