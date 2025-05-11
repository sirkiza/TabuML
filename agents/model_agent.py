import json
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

def run_model_agent():
    messages = [
        {"role": "system", "content": "You are a machine learning assistant that helps to train machine learning models."},
        {"role": "user", "content": "Please train a model on an income dataset data/adult.csv. Try to pick different parameters to maximize accuracy (desired accuracy is above 88%). Output the final result as a formatted table and add reasoning behind your choices. Make at least 10 iterations but not more than 20 iterations"} 
    ]
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="o4-mini",  # Must support function calling
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    dataset_path = "data/adult.csv"
    target_column = "income"
    dataset_name = "adult"
    
    i = 0
    while response.choices[0].finish_reason != "stop":
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            function_call = choice.message.tool_calls[0].function
            call_id = choice.message.tool_calls[0].id
            arguments = json.loads(function_call.arguments)
            print(f"Iteration {i}: calling function {function_call.name} with arguments {function_call.arguments}")
            
            # Step 1: Load dataset
            X, y, meta = load_dataset(dataset_path, target_column)
            
            if function_call.name == "train_random_forest":
                # Step 2: Preprocess (model-aware)
                X_proc, y_proc, pipeline = preprocess_data(X, y, "RandomForest")
                
                config = {
                        "algorithm": "RandomForest",
                        "params": {
                            "n_estimators": arguments["n_estimators"],
                            "max_depth": arguments["max_depth"]
                        }
                    }

                
            if function_call.name == "train_logistic_regression":
                # Step 2: Preprocess (model-aware)
                X_proc, y_proc, pipeline = preprocess_data(X, y, "LogisticRegression")
                
                config = {
                        "algorithm": "LogisticRegression",
                        "params": {
                            "penalty": arguments["penalty"],
                            "max_iter": arguments["max_iter"],
                            "solver": arguments["solver"]
                        }
                    }
            if function_call.name == "train_xgboost":
                # Step 2: Preprocess (model-aware)
                X_proc, y_proc, pipeline = preprocess_data(X, y, "XGBoost")
                
                config = {
                        "algorithm": "XGBoost",
                        "params": {
                            "n_estimators": arguments["n_estimators"],
                            "learning_rate": arguments["learning_rate"],
                            "max_depth": arguments["max_depth"],
                            "subsample": arguments["subsample"]
                        }
                    }
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_proc, y_proc, test_size=0.2, random_state=42
            )
                
            # Step 3: Evaluate
            results = run_evaluation_agent(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_config=config,
                dataset_name=dataset_name
            )
            
            print(f"Iteration {i} results: accuracy {results['accuracy']}")

            messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function_call.name,
                            "arguments": function_call.arguments
                        }
                    }
                ]
            }
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(results)
            })
            
            response = client.chat.completions.create(
                model="o4-mini",  # Must support function calling
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            i += 1
                
    return response.choices[0].message.content
    
    