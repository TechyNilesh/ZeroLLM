"""Server example — expose a local LLM as an OpenAI-compatible API."""

from zerollm import Server

# One line to serve
Server("Qwen/Qwen3.5-4B", port=8080).serve()

# Then use with any OpenAI-compatible client:
#
#   curl http://localhost:8080/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "Hello!"}]}'
#
# Or with the OpenAI Python client:
#
#   from openai import OpenAI
#   client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
#   response = client.chat.completions.create(
#       model="Qwen/Qwen3.5-4B",
#       messages=[{"role": "user", "content": "Hello!"}],
#   )
