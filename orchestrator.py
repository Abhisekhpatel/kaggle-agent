import json
import subprocess
from pathlib import Path
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def kaggle_cmd(args):
    result = subprocess.run(["kaggle"] + args, capture_output=True, text=True)
    return result.stdout + result.stderr

TOOLS = [
    {"name": "list_competitions", "description": "List active Kaggle competitions.", "input_schema": {"type": "object", "properties": {"category": {"type": "string", "default": "featured"}}}},
    {"name": "download_competition", "description": "Download competition data.", "input_schema": {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}}
]

def execute_tool(name, inputs):
    if name == "list_competitions":
        return kaggle_cmd(["competitions", "list"])
    elif name == "download_competition":
        Path("data").mkdir(exist_ok=True)
        return kaggle_cmd(["competitions", "download", "-c", inputs["slug"], "-p", "data", "--unzip"])
    return "unknown tool"

messages = [{"role": "user", "content": "List active Kaggle competitions and pick the best tabular one for a beginner."}]

while True:
    response = client.messages.create(model="claude-opus-4-8", max_tokens=2048, tools=TOOLS, messages=messages)
    for block in response.content:
        if hasattr(block, "text"):
            print(block.text)
    if response.stop_reason == "end_turn":
        break
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[Tool] {block.name}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": execute_tool(block.name, block.input)})
        messages.append({"role": "user", "content": results})
