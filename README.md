# End-to-End MLOps & Federated Learning Fraud Detection
Bu proje, İstanbul Atlas Üniversitesi Bilgisayar Mühendisliği bitirme projesi kapsamında geliştirilmektedir.

## Proje Amacı
Kredi kartı dolandırıcılığı tespitini, bankalar arasında veri gizliliğini koruyarak (Federated Learning) ve tüm süreci MLOps prensipleriyle izleyerek (MLflow) gerçekleştirmektir. 

## 🛠️ Kullanılan Teknolojiler
- Federated Learning: Flower (flwr) 
- Experiment Tracking: MLflow 
- Model: Scikit-learn Logistic Regression
- Data: Kaggle Credit Card Fraud Detection Dataset

# --------------------------------------------------------------------

## 🚀 Projeyi Docker ile Canlıya Alma (Deployment)

Tahmin için `FRAUD_MODEL_PATH` ile güvenilir bir model ve eşleşen ön işleme
meta verisi sağlanmalıdır. Mevcut `best_model.pkl` için doğrulanmış meta veri
yoktur. Kurulum ayrıntıları için [Phase 1B preprocessing](docs/PREPROCESSING.md).

Bu proje, eğitilmiş Federe Öğrenme (Federated Learning) modelini mikroservis mimarisine uygun olarak dış dünyaya açmak için **FastAPI** ve **Docker** kullanmaktadır. Projeyi herhangi bir ortamda tek tıkla çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### Ön Koşullar
* Bilgisayarınızda **Docker** yüklü ve çalışır durumda olmalıdır.

### Çalıştırma Adımları

**1. Docker İmajını İnşa Edin**
Projenin ana dizininde (Dockerfile'ın bulunduğu konumda) terminali açın ve aşağıdaki komutu çalıştırarak projeyi paketleyin:
`docker build -t fraud-api .`

**2. Konteyneri Ayağa Kaldırın**
İnşa işlemi tamamlandıktan sonra, API'ı arka planda 8000 portunda çalıştırmak için:
`docker run -d -p 8000:8000 --name bitirme-api fraud-api`

### API'ı Test Etme
Servis çalışmaya başladıktan sonra tarayıcınızdan interaktif API dokümantasyonuna erişebilirsiniz:
👉 **http://localhost:8000/docs**

Bu arayüz üzerinden `/predict` endpoint'ine tıklayarak test verilerinizi JSON formatında gönderebilir ve modelin canlı tahminlerini (`is_fraud: true` veya `false`) anında görebilirsiniz. 

**Örnek İstek Formatı:**
```json
{
  "features": [0.0, -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 149.62]
}
```

## Phase 0: reproducible local environment

Use **64-bit CPython 3.12.10** and a new `.venv-phase0` environment. The existing
virtual environments are retained. Install the hash-checked `requirements.lock`
for reproducibility; `requirements.txt` lists the direct dependency pins.

- [Exact setup and dependency decisions](docs/ENVIRONMENT.md)
- [Dataset acquisition, expected schema, and fingerprint](docs/DATASET.md)
- [Executed Phase 0 checks and remaining limitations](docs/PHASE0_RESULTS.md)

This phase changes environment setup only. The earlier audit's training and
inference correctness issues remain open. The Docker instructions above still use
the original Dockerfile and are not the validated Phase 0 environment.
