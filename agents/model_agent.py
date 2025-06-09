import json
from sklearn.model_selection import train_test_split

from tools.data_preprocessor import preprocess_data
from agents.evaluation_agent import run_evaluation_agent

ROLE_PROMPT = """
    You are a machine learning assistant named TabuML that helps to train machine learning models with AutoML for tabular data classification. 
    Your task is model selection and evaluation. You are provided the dataset and a set of classification tools you can call. Follow the
    instructions in the user prompt and perform multiple iterations to achieve the desired result. Experiment with hyperparameters to achieve 
    the desired result. Return your responses formatted as HTML.
"""

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
    },
    {
        "type": "function",
        "function": {
                "name": "train_lightgbm",
                "description": "Train a model using LightGBM with specified parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_leaves": {
                            "type": "integer",
                            "description": "Maximum number of leaves in one tree. Controls model complexity."
                        },
                        "learning_rate": {
                            "type": "number",
                            "description": "Boosting learning rate. Smaller values prevent overfitting but require more iterations."
                        },
                        "n_estimators": {
                            "type": "integer",
                            "description": "Number of boosting iterations (trees)."
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth of a tree. Used to control overfitting. -1 means no limit."
                        },
                        "subsample": {
                            "type": "number",
                            "description": "Fraction of data used for training each tree (0.0 to 1.0). Helps prevent overfitting."
                        }
                    },
                    "required": ["num_leaves", "learning_rate", "n_estimators", "max_depth", "subsample"]
                }
        }
    }
]


async def run_model_agent(client, chat, user_prompt, X, y):
    print("Running model agent")
    messages = [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="o4-mini",  # Must support function calling
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    i = 0
    while response.choices[0].finish_reason != "stop":
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            function_call = choice.message.tool_calls[0].function
            call_id = choice.message.tool_calls[0].id
            arguments = json.loads(function_call.arguments)

            if function_call.name == "train_random_forest":
                # Step 2: Preprocess (model-aware)
                X_proc, y_proc, pipeline = preprocess_data(
                    X, y, "RandomForest")

                config = {
                    "algorithm": "RandomForest",
                    "params": {
                        "n_estimators": arguments["n_estimators"],
                        "max_depth": arguments["max_depth"]
                    }
                }

            if function_call.name == "train_logistic_regression":
                # Step 2: Preprocess (model-aware)
                X_proc, y_proc, pipeline = preprocess_data(
                    X, y, "LogisticRegression")

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

            if function_call.name == "train_lightgbm":
                # Step 2: Preprocess (model-aware)
                X_proc, y_proc, pipeline = preprocess_data(X, y, "LightGBM")

                config = {
                    "algorithm": "LightGBM",
                    "params": {
                        "num_leaves": arguments["num_leaves"],
                        "learning_rate": arguments["learning_rate"],
                        "n_estimators": arguments["n_estimators"],
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
                model_config=config
            )

            await chat.iteration(i, config['algorithm'], config['params'], results['report'])

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
                "content": json.dumps(results['report'])
            })

            response = client.chat.completions.create(
                model="o4-mini",  # Must support function calling
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            i += 1
    
    print("Model agent finished")

    return response.choices[0].message.content
