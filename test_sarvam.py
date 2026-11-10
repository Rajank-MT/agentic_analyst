import os

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY is not configured.")

client = SarvamAI(
    api_subscription_key=api_key
)

response = client.chat.completions(
    model="sarvam-105b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Sarvam integration successful"
        }
    ],
    temperature=0.1,
    max_tokens=100,
)

print("SARVAM RESPONSE:")
print(response.choices[0].message.content)