# intitilizing dependencies
import sys
from openai import OpenAI
import json
from pathlib import Path

# initilizing local llm
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gemma4:e4b-32k"

WORKSPACE_DIR = Path(__file__).parent / "workspace"

print(f"Workspace directory: {WORKSPACE_DIR}")

# ======================================
# tools
# ======================================


def list_files() -> str:
    res = ""
    # check if workspace is empty
    if not any(WORKSPACE_DIR.iterdir()):
        return "Workspace is empty."
    for dir in WORKSPACE_DIR.iterdir():
        res += dir.name + "\n"
    return res


def read_file(file_name: str) -> str:
    file_path = WORKSPACE_DIR / file_name
    if not file_path.exists():
        return f"File not found: {file_name}"
    return file_path.read_text(encoding="utf-8")


def write_file(file_name: str, content: str) -> str:
    """This will write to the file in the workspace"""
    (WORKSPACE_DIR/file_name).write_text(content, encoding="utf-8")
    return f"Wrote ${file_name}"


# List of tools
TOOLS = {
    "list_files": list_files,
    "read_files": read_file,
    "write_file": write_file
}

# Tool Schema
# lmma cannot call tools they need schema
TOOLS_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the file in users workspace folder",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": "Read one file from user's workspace folder",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
                "name": "write_file",
                "description": "Write one file from user's workspace folder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["file_name", "content"]
                }
        }
    }
]


# chat function
def chat(user_message: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
        # it will keep on calling tool the at the end call the messages
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMAS
            )
            message = response.choices[0].message

            
            if not message.tool_calls:
                return message.content or ""  # model did not call any tool, return the message
            
            # model needs to call tools
            messages.append(message)
            # calling tool on last message
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or {})
                print(f"    [tool] {call.function.name} {args}")
                result = TOOLS[call.function.name](**args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result
                    }
                )
    except Exception as e:
        return f"An error occurred: {e}"


# main
if __name__ == "__main__":
    print(
        f"Welcome to the local {MODEL} LLM chat interface. Type'exit' to quit.\n\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chat. Goodbye!")
                break
            response = chat(user_input)
            print("\n\n")
            print(f"Assistant: {response}")
            print("\n\n")
        except KeyboardInterrupt:
            print("\nExiting the chat. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print("error occor")
            print(e)
