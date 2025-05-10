from agents.prompt_agent import run_prompt_agent
from agents.model_agent import run_model_agent
from evaluation.evaluation_agent import load_dataset, run_evaluation_agent

def main():
    prompt = 'Predict if a person earns more than \ based on their demographics'
    parsed = run_prompt_agent(prompt)
    model_config = run_model_agent(parsed['task_type'])
    X_train, X_test, y_train, y_test = load_dataset(target_col=parsed['target_column'])
    results = run_evaluation_agent(X_train, X_test, y_train, y_test, model_config)

    with open('output/report.md', 'w') as f:
        f.write(f"# AutoML Baseline Report\n\n")
        f.write(f"**Accuracy**: {results['accuracy']:.2f}\n\n")
        f.write("## Classification Report\n")
        f.write(f"`\n{results['report']}\n`\n")
        f.write("## Explanation\n")
        f.write("Selected RandomForest for its ability to handle mixed-type tabular data robustly.")

if __name__ == '__main__':
    main()
