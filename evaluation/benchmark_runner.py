import pandas as pd
from evaluation.dataset_loader import load_dataset
from evaluation.data_preprocessor import preprocess_data
from evaluation.evaluation_agent import run_evaluation_agent

# List of datasets to run (extend as needed)
DATASETS = [
    {
        "name": "adult",
        "path": "data/adult.csv",
        "target_column": "income",
        "model_config": {
            "algorithm": "RandomForest",
            "params": {"n_estimators": 100, "max_depth": 10}
        }
    },
]

def run_benchmark():
    results = []

    for entry in DATASETS:
        print(f"\n📊 Running benchmark on: {entry['name']}")

        try:
            X, y, meta = load_dataset(entry["path"], entry["target_column"])
            X_proc, y_proc, pipeline = preprocess_data(X, y, model_type=entry["model_config"]["algorithm"])
            result = run_evaluation_agent(
                X_train=X_proc,
                X_test=X_proc,  # For now, using full data to test pipeline; replace with proper split if needed
                y_train=y_proc,
                y_test=y_proc,
                model_config=entry["model_config"],
                dataset_name=entry["name"]
            )

            # Add dataset metadata to results
            result.update({
                "num_rows": meta["num_rows"],
                "num_columns": meta["num_columns"],
                "num_cat": meta["num_categorical"],
                "num_num": meta["num_numerical"]
            })

            results.append(result)

        except Exception as e:
            print(f"❌ Failed on {entry['name']}: {e}")
            continue

    # Save results table
    df = pd.DataFrame(results)
    df.to_csv("output/benchmark_results.csv", index=False)
    print("\n✅ Benchmark complete. Results saved to output/benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
