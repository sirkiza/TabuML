import json
from tools.dataset_loader import load_dataset


ROLE_PROMPT = """
    You are a machine learning assistant named TabuML that helps to train machine learning models with AutoML for tabular data classification. 
    Your task is loading and data preprocessing of a dataset. You will be given tools to load the dataset and execute the preprocessing.
"""

tools = [
    {
        "type": "function",
        "function": {
                "name": "load_dataset",
                "description": "Load dataset from local filesystem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the dataset file in local filesystem"
                        },
                        "target": {
                            "type": "string",
                            "description": "target column"
                        }
                    },
                    "required": ["path", "target"]
                }
        }
    }
]


async def run_data_processing_agent(client, chat, user_prompt):
    print("Running data processing agent")
    messages = [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    X, y = None, None

    while response.choices[0].finish_reason != "stop":
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            function_call = choice.message.tool_calls[0].function
            call_id = choice.message.tool_calls[0].id
            arguments = json.loads(function_call.arguments)

            if function_call.name == "load_dataset":
                X, y, metadata = load_dataset(
                    arguments['path'], arguments['target'])
                await chat.dataset_metadata(metadata)
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
                    "content": "Loaded successfully"
                })

                response = client.chat.completions.create(
                    model="o4-mini",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
    
    print("Data processing agent finished")

    return X, y
