from glob import escape
import json


class UserChat:
    def __init__(self, websocket):
        self.websocket = websocket

    async def ask_question(self, question):
        await self.websocket.send_text(json.dumps({
            "text": question,
            "message_type": "question"
        }))

    async def dataset_metadata(self, metadata):
        await self.websocket.send_text(json.dumps({
            "text": f"<h3>Dataset metadata</h3>{json_to_html_table(metadata)}",
            "message_type": "info"
        }))
    
    async def iteration(self, iteration_number, method, arguments, report):
        await self.websocket.send_text(json.dumps({
            "text": f"""
                <h3>Iteration #{iteration_number}</h3> 
                <h4>Used <b>{method}</b> with arguments:</h4> 
                {json_to_html_table(arguments)} </br></br> 
                <h4>Results</h4>
                {json_to_html_table(report)}
            """,
            "message_type": "info"
        }))
        
    async def report(self, report):
        await self.websocket.send_text(json.dumps({
            "text": report,
            "message_type": "report"
        }))
    
    async def input(self):
        return await self.websocket.receive_text()

def json_to_html_table(data):
    def render(value):
        if isinstance(value, dict):
            return json_to_html_table(value)
        elif isinstance(value, list):
            return "<ul>" + "".join(f"<li>{render(v)}</li>" for v in value) + "</ul>"
        else:
            return escape(str(value))

    html = "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; font-family: sans-serif;'>"
    for key, value in data.items():
        html += "<tr>"
        html += f"<th style='text-align: left; background: #f0f0f0;'>{escape(str(key))}</th>"
        html += f"<td>{render(value)}</td>"
        html += "</tr>"
    html += "</table>"
    return html