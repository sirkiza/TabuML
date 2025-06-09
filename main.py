import sys
import warnings
from sklearn.exceptions import ConvergenceWarning

from agents.model_agent import run_model_agent

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <dataset_path> <target_column>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    target_column = sys.argv[2]
    dataset_name = dataset_path.split("/")[-1].split(".")[0]

    result = run_model_agent(dataset_path, target_column, dataset_name)

    print("\n===== Best Result Summary =====")
    print(f"Accuracy: {result.get('accuracy')}")
    print(f"Training Duration: {result.get('training_duration')}s")
    print(f"LLM Reasoning:\n{result.get('llm_reasoning')}\n")

if __name__ == "__main__":
    main()
