import json
import time
from openai import OpenAI
from sklearn.model_selection import train_test_split

from evaluation.data_preprocessor import preprocess_data
from evaluation.dataset_loader import load_dataset
from evaluation.evaluation_agent import run_evaluation_agent


tools = [
        {
            "type": "function",
            "function": {
                "name": "train_random_forest",
                "description": "Train a model using random forest with parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n_estimators": {
                            "type": "integer",
                            "description": "Number of estimators"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "maximal depth of the tree"
                        }
                    },
                    "required": ["n_estimators", "max_depth"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "train_logistic_regression",
                "description": "Train a model using logistic regression with parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "penalty": {
                            "type": "string",
                            "enum": ["l1", "l2"],
                            "description": "Form of the penalty"
                        },
                        "max_iter": {
                            "type": "integer",
                            "description": "Max number of iterations"
                        },
                        "solver": {
                            "type": "string",
                            "enum": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                            "description": "Form of the penalty"
                        }
                    },
                    "required": ["penalty", "max_iter", "solver"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "train_xgboost",
                "description": "Train a model using xgboost with parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n_estimators": {
                            "type": "integer",
                            "description": "Number of boosting rounds (trees)"
                        },
                        "learning_rate": {
                            "type": "number",
                            "description": "Step size shrinkage to prevent overfitting. Smaller = slower but more accurate."
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max tree depth. Larger = more complex models."
                        },
                        "subsample": {
                            "type": "number",
                            "description": "Fraction of training data used per tree. Lower to prevent overfitting"
                        }
                    },
                    "required": ["penalty", "max_iter", "solver"]
                }
            }
        }
    ]

def run_model_agent(dataset_path, target_column, dataset_name):
    system_prompt = (
        "You are a machine-learning assistant. "
        "Try to explore the hyperparameter space and maximize validation accuracy. "
        "Use at least 10 but not more than 20 trials."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Dataset path: {dataset_path}\n"
                f"Target column: {target_column}\n"
                f"Please begin hyper-parameter exploration."
            ),
        },
    ]

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    best_accuracy = -1
    best_output = None
    iteration = 0

    while response.choices[0].finish_reason != "stop":
        choice = response.choices[0]
        if choice.finish_reason != "tool_calls":
            break

        call = choice.message.tool_calls[0]
        func_name = call.function.name
        args = json.loads(call.function.arguments)
        call_id = call.id

        print(f"Iteration {iteration}: {func_name}  params={args}")

        X, y, _ = load_dataset(dataset_path, target_column)

        if func_name == "train_random_forest":
            model_type = "RandomForest"
        elif func_name == "train_logistic_regression":
            model_type = "LogisticRegression"
        elif func_name == "train_xgboost":
            model_type = "XGBoost"
        else:
            print(f"[Warning] Unsupported function {func_name}")
            continue

        X_proc, y_proc, _ = preprocess_data(X, y, model_type)
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y_proc, test_size=0.2, random_state=42)
        model_config = {"algorithm": model_type, "params": args}
        result_dict = run_evaluation_agent(X_train, X_test, y_train, y_test, model_config, dataset_name=dataset_name)

        acc = result_dict["accuracy"]
        print(f"→ accuracy={acc:.4f}")

        if len(result_dict["predictions"]) == len(y_test) and acc > best_accuracy:
            best_accuracy = acc
            best_output = {
                "predictions": list(result_dict["predictions"]),
                "truth": y_test.tolist(),
                "probabilities": result_dict.get("probabilities"),
                "probabilities_labels": result_dict.get("probabilities_labels"),
                "training_duration": result_dict.get("train_time_sec", 0.0),
                "accuracy": acc,
                "llm_reasoning": None,
            }

        messages += [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result_dict)
            }
        ]

        response = client.chat.completions.create(
            model="o4-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        iteration += 1

    if best_output:
        best_output["llm_reasoning"] = response.choices[0].message.content

    return best_output
