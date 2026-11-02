import sys
from openai import OpenAI

# initilizing local llm

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gemma4:e4b-32k"


def chat(user_message: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print(f"Welcome to the local {MODEL} LLM chat interface. Type'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chat. Goodbye!")
                break
            response = chat(user_input)
            print(f"Assistant: {response}")
        except KeyboardInterrupt:
            print("\nExiting the chat. Goodbye!")
            sys.exit(0)
