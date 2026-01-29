#!/usr/bin/env python3
"""
Exercise 1 Solution: Make your first LLM API call
=================================================
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Create the OpenAI client (uses OPENAI_API_KEY from environment)
client = OpenAI()


def ask_llm(question: str) -> str:
    """
    Send a question to the LLM and return the response.

    Args:
        question: The question to ask

    Returns:
        The LLM's response text
    """
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0.7
    )

    # Extract the text from the response
    return response.choices[0].message.content


# Test the function
if __name__ == "__main__":
    # Simple test question
    question = "What does a Kubernetes pod do? Answer in one sentence."

    print("Making API call to LLM...")
    response = ask_llm(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")

    # Let's try another question
    print("\n" + "=" * 50 + "\n")

    question2 = "What is Terraform used for? Keep it brief."
    response2 = ask_llm(question2)

    print(f"Question: {question2}")
    print(f"Answer: {response2}")
