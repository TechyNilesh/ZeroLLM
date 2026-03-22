"""Structured JSON output — get valid JSON from the LLM."""

from zerollm import Chat

bot = Chat("Qwen/Qwen3.5-4B")

# Generate structured JSON with a schema
result = bot.backend.generate_json(
    messages=[
        {"role": "user", "content": "List 3 programming languages with their year of creation"}
    ],
    schema={
        "type": "object",
        "properties": {
            "languages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "year": {"type": "integer"},
                    },
                },
            },
        },
    },
)

print(result)
# {"languages": [{"name": "Python", "year": 1991}, ...]}

# Without a schema — just get valid JSON
result = bot.backend.generate_json(
    messages=[{"role": "user", "content": "Give me a JSON object with today's weather in Auckland"}]
)
print(result)
