import flwr as fl
import mlflow
import os
import argparse

parser = argparse.ArgumentParser(description="Flower Server")
parser.add_argument("--num-clients", type=int, required=True, help="Beklenen toplam banka sayısı")
args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(os.path.dirname(current_dir), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("Credit_Card_Fraud_Federated_Learning")

class MLflowStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        print(f"\n>>> Tur {server_round} değerlendiriliyor...")
        
        if metrics and "accuracy" in metrics:
            acc = metrics["accuracy"]
            print(f">>> Başarı Oranı: {acc} - MLflow'a yazılıyor...")
            
            try:
                with mlflow.start_run(run_name=f"Round_{server_round}"):
                    mlflow.log_metric("accuracy", acc, step=server_round)
                    mlflow.log_metric("loss", loss if loss is not None else 0.0, step=server_round)
                print(">>> MLflow yazma işlemi başarıyla tamamlandı.")
            except Exception as e:
                print(f">>> MLflow HATASI: {e}")
        return loss, metrics

strategy = MLflowStrategy(
    fraction_fit=1.0,
    min_fit_clients=args.num_clients,
    min_evaluate_clients=args.num_clients,
    min_available_clients=args.num_clients,
    evaluate_metrics_aggregation_fn=lambda metrics: {"accuracy": sum([m[1]["accuracy"] for m in metrics]) / len(metrics)}
)

if __name__ == "__main__":
    print(f"MLflow DB Yolu: {db_path}")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )