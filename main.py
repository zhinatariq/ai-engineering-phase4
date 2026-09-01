import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def should_search(question: str) -> bool:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are deciding whether to search the course "
                    "documentation for a Data Structures study tool.\n\n"
                    "The course documentation is the authoritative source "
                    "for answers about Data Structures.\n\n"
                    "Even if you already know the answer from general "
                    "knowledge, choose SEARCH when the question is related "
                    "to topics that may be covered in the course material.\n\n"
                    "Choose DIRECT only when the question is clearly "
                    "unrelated to the course documentation and can be "
                    "answered without it.\n\n"
                    "Respond with exactly one word:\n"
                    "SEARCH or DIRECT\n\n"
                    "Output only SEARCH or DIRECT."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    decision = response.choices[0].message.content.strip().lower()

    print(f"Model decision: {decision}")

    # Only an explicit DIRECT decision skips retrieval.
    # Anything unexpected safely defaults to SEARCH.
    if "direct" in decision:
        return False

    return True


# Manual tests
print("Stack:", should_search("What is a stack?"))
print("2 + 2:", should_search("What's 2+2?"))
print(
    "Ambiguous:",
    should_search("Can I use a stack for something other than what's in the docs?")
)
