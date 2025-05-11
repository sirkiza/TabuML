from sklearn.exceptions import ConvergenceWarning
from agents.model_agent import run_model_agent
from evaluation.dataset_loader import load_dataset
from evaluation.data_preprocessor import preprocess_data
from evaluation.evaluation_agent import run_evaluation_agent
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    result = run_model_agent()
    
    print()
    print(result)

if __name__ == "__main__":
    main()
