import json
import pandas as pd
from sklearn.model_selection import train_test_split
from openai import OpenAI

from evaluation.data_preprocessor import preprocess_data
from evaluation.evaluation_agent import run_evaluation_agent


class TabuMLAutoML:
    def __init__(self, dataset_name: str = "unknown"):
        self.dataset_name = dataset_name
        self.client = OpenAI()
        self.model_output = None
        self.best_model = None
        self.model_type = None
        self.preprocessor = None
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a machine-learning assistant. "
                    "Try to explore the hyperparameter space and maximize validation accuracy. "
                    "Use at least 10 but not more than 20 trials."
                ),
            }
        ]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "train_random_forest",
                    "description": "Train a model using random forest with parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_estimators": {"type": "integer"},
                            "max_depth": {"type": "integer"},
                        },
                        "required": ["n_estimators", "max_depth"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "train_logistic_regression",
                    "description": "Train a model using logistic regression with parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "penalty": {"type": "string", "enum": ["l1", "l2"]},
                            "max_iter": {"type": "integer"},
                            "solver": {
                                "type": "string",
                                "enum": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                            },
                        },
                        "required": ["penalty", "max_iter", "solver"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "train_xgboost",
                    "description": "Train a model using xgboost with parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_estimators": {"type": "integer"},
                            "learning_rate": {"type": "number"},
                            "max_depth": {"type": "integer"},
                            "subsample": {"type": "number"},
                        },
                        "required": ["n_estimators", "learning_rate", "max_depth", "subsample"],
                    },
                },
            },
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.messages.append({
            "role": "user",
            "content": f"Target column: {y.name or 'target'}\nPlease begin hyper-parameter exploration."
        })

        best_accuracy = -1
        best_output = None
        iteration = 0

        response = self.client.chat.completions.create(
            model="o4-mini",
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto"
        )

        while response.choices[0].finish_reason != "stop":
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls":
                break

            call = choice.message.tool_calls[0]
            func_name = call.function.name
            args = json.loads(call.function.arguments)
            call_id = call.id

            print(f"Iteration {iteration}: {func_name}  params={args}")

            model_type = {
                "train_random_forest": "RandomForest",
                "train_logistic_regression": "LogisticRegression",
                "train_xgboost": "XGBoost",
            }.get(func_name)

            if not model_type:
                print(f"[Warning] Unsupported function {func_name}")
                continue

            # Re-preprocess for this model type
            X_proc, y_proc, preprocessor = preprocess_data(X, y, model_type=model_type)
            X_train, X_val, y_train, y_val = train_test_split(X_proc, y_proc, test_size=0.2, random_state=42)

            model_config = {"algorithm": model_type, "params": args}
            result_dict = run_evaluation_agent(X_train, X_val, y_train, y_val, model_config, dataset_name=self.dataset_name)

            acc = result_dict.get("accuracy", 0)
            print(f"→ accuracy={acc:.4f}")

            if len(result_dict["predictions"]) == len(y_val) and acc > best_accuracy:
                best_accuracy = acc
                self.best_model = result_dict["model"]
                self.preprocessor = preprocessor

                best_output = {
                    "predictions": result_dict["predictions"],
                    "truth": y_val.tolist(),
                    "probabilities": result_dict.get("probabilities"),
                    "probabilities_labels": result_dict.get("probabilities_labels"),
                    "training_duration": result_dict.get("train_time_sec", 0.0),
                    "accuracy": acc,
                    "model_type": model_type,
                    "hyperparameters": args,
                    "selected_iteration": iteration,
                    "llm_reasoning": None,
                }

            self.messages += [
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
                    "content": json.dumps({
                        "accuracy": acc,
                        "model_type": model_type,
                        "status": "success"
                    })
                }
            ]

            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto"
            )
            iteration += 1

        if best_output:
            best_output["llm_reasoning"] = response.choices[0].message.content

        self.model_output = best_output

    def predict(self, X: pd.DataFrame):
        if self.best_model is None or self.preprocessor is None:
            raise RuntimeError("You must call .fit() before .predict().")

        X_proc = self.preprocessor.transform(X)

        predictions = self.best_model.predict(X_proc)
        probabilities = (
            self.best_model.predict_proba(X_proc)
            if hasattr(self.best_model, "predict_proba")
            else None
        )
        return predictions, probabilities
