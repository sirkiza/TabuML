from agents.data_processing_agent import run_data_processing_agent
from agents.model_agent import run_model_agent
from agents.prompt_agent import run_prompt_agent

async def run_coordinator(client, user_chat):
    prompt = await run_prompt_agent(client, user_chat)
    X, y = await run_data_processing_agent(client, user_chat, prompt)
    result = await run_model_agent(client, user_chat, prompt, X, y)
    await user_chat.report(result)
    