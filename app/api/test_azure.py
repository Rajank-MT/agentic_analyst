
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

required = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]

missing = [name for name in required if not os.getenv(name)]
if missing:
    raise RuntimeError(
        f"Missing .env variables: {', '.join(missing)}"
    )

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with: Azure connection successful."},
    ],
    max_completion_tokens=100,
)

print(response.choices[0].message.content)
