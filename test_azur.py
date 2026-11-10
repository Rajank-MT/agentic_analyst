
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    AzureOpenAI,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

# Load .env from the same directory as this Python file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_required_env(name: str) -> str:
    """Read a required environment variable and validate it."""
    value = os.getenv(name, "").strip()

    if not value:
        raise ValueError(
            f"Missing environment variable: {name}\n"
            f"Please configure it in: {env_path}"
        )

    if value.startswith("your-") or "YOUR-RESOURCE" in value:
        raise ValueError(
            f"{name} still contains a placeholder. "
            "Replace it with the actual Azure value."
        )

    return value


def main():
    try:
        endpoint = get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
        api_key = get_required_env("AZURE_OPENAI_API_KEY")
        api_version = get_required_env("AZURE_OPENAI_API_VERSION")
        deployment = get_required_env("AZURE_OPENAI_DEPLOYMENT")

    except ValueError as exc:
        print(f"\nCONFIGURATION ERROR:\n{exc}")
        sys.exit(1)

    # Print configuration safely. Never print the API key.
    print("=" * 55)
    print("Azure OpenAI Connection Test")
    print("=" * 55)
    print(f"Endpoint:       {endpoint}")
    print(f"API version:    {api_version}")
    print(f"Deployment:     {deployment}")
    print(f".env file:      {env_path}")
    print("=" * 55)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=30.0,
        max_retries=1,
    )

    try:
        print("\nSending test request to Azure OpenAI...")

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly: "
                        "Azure connection successful."
                    ),
                },
            ],
            max_completion_tokens=100,
        )

        answer = response.choices[0].message.content

        print("\nSUCCESS!")
        print(f"Response: {answer}")
        print(f"Model returned: {response.model}")

    except AuthenticationError:
        print("\nAUTHENTICATION ERROR (401)")
        print("Check AZURE_OPENAI_API_KEY.")
        print("Copy a valid key from the same Azure resource.")

    except RateLimitError as exc:
        print("\nRATE LIMIT OR QUOTA ERROR (429)")
        print("Check your model deployment quota and limits.")
        print(f"Details: {exc}")

    except APIConnectionError as exc:
        print("\nCONNECTION ERROR")
        print("Check your internet connection and endpoint.")
        print(f"Details: {exc}")

    except APIStatusError as exc:
        print(f"\nAZURE API ERROR: HTTP {exc.status_code}")
        print(f"Error details: {exc}")

        if exc.status_code == 404:
            print("\nPossible causes:")
            print("1. The deployment name is incorrect.")
            print("2. The endpoint belongs to another resource.")
            print("3. The model deployment has not finished.")
            print("\nVerify the deployment name and endpoint in Foundry.")

    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {type(exc).__name__}")
        print(str(exc))

    finally:
        client.close()


if __name__ == "__main__":
    main()
