from evaluation.dataset_loader import load_dataset
from evaluation.data_preprocessor import preprocess_data
from evaluation.evaluation_agent import run_evaluation_agent
import json
import os

def main():
    # Config
    dataset_path = "data/adult.csv"
    target_column = "income"
    model_config = {
        "algorithm": "RandomForest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }
    dataset_name = "adult"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load dataset
    X, y, meta = load_dataset(dataset_path, target_column)

    # Step 2: Preprocess (model-aware)
    X_proc, y_proc, pipeline = preprocess_data(X, y, model_config["algorithm"])

    # Step 3: Evaluate
    results = run_evaluation_agent(
        X_train=X_proc,
        X_test=X_proc,  # For now, use full dataset (or plug in split here)
        y_train=y_proc,
        y_test=y_proc,
        model_config=model_config,
        dataset_name=dataset_name
    )

    # Step 4: Print and save
    print("\n✅ Evaluation Results:")
    print("Accuracy:", results["accuracy"])
    print("Training Time (s):", results["train_time_sec"])

    report_path = os.path.join(output_dir, f"{dataset_name}_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n📄 Full report saved to: {report_path}")

if __name__ == "__main__":
    main()
