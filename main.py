from sklearn.exceptions import ConvergenceWarning
from agents.model_agent import run_model_agent
from evaluation.dataset_loader import load_dataset
from evaluation.data_preprocessor import preprocess_data
from evaluation.evaluation_agent import run_evaluation_agent
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    import sys

    if len(sys.argv) != 3:
        print("Usage: python main.py <dataset_path> <target_column>")
        return

    dataset_path = sys.argv[1]
    target_column = sys.argv[2]
    dataset_name = dataset_path.split("/")[-1].split(".")[0]

    result = run_model_agent(dataset_path, target_column, dataset_name)

    print()
    print(result)

if __name__ == "__main__":
    main()
