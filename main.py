from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import uvicorn

from agents.coordinator import run_coordinator
from portal.user_chat import UserChat

# INSERT API KEY HERE
OPENAI_KEY = ""

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat = UserChat(websocket)
    client = OpenAI(api_key=OPENAI_KEY)
    await run_coordinator(client, chat)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
