import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("foodnet_model_comparison")

results = [
    {
        "model_name": "MobileNetV2",
        "architecture": "transfer_learning",
        "params": 2_600_000,
        "test_accuracy": 0.9938,
        "epochs": 25,
        "input_size": "100x100",
        "artifact": "results/training_curves.png"
    },
    {
        "model_name": "VGG16",
        "architecture": "transfer_learning",
        "params": 14_700_000,
        "test_accuracy": 0.9938,
        "epochs": 15,
        "input_size": "100x100",
        "artifact": "results/model_comparison.png"
    },
    {
        "model_name": "CustomCNN",
        "architecture": "from_scratch",
        "params": 4_863_023,
        "test_accuracy": 0.9887,
        "epochs": 15,
        "input_size": "100x100",
        "artifact": "results/custom_cnn_confusion_matrix.png"
    },
]

for r in results:
    try:
        with mlflow.start_run(run_name=r["model_name"]):
            mlflow.log_param("architecture", r["architecture"])
            mlflow.log_param("epochs", r["epochs"])
            mlflow.log_param("input_size", r["input_size"])
            mlflow.log_param("total_params", r["params"])
            mlflow.log_param("random_seed", 42)
            mlflow.log_metric("test_accuracy", r["test_accuracy"])
            mlflow.log_artifact(r["artifact"])
        print(f"Logged {r['model_name']} successfully")
    except Exception as e:
        print(f"FAILED to log {r['model_name']}: {e}")

print("Done.")