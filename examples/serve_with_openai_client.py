"""Use ZeroLLM server with OpenAI's Python client.

First start the server:
    zerollm serve Qwen/Qwen3.5-4B --port 8080

Then run this script to connect with the OpenAI client.
"""

from openai import OpenAI

# Point the OpenAI client to your local ZeroLLM server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="unused",  # ZeroLLM doesn't require an API key
)

# Chat completion — same API as OpenAI
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of New Zealand?"},
    ],
)
print(response.choices[0].message.content)

# Streaming
print("\nStreaming:")
stream = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
