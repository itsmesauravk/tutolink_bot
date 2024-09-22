import openai
import os

# Load the API key from an environment variable
openai.api_key = "api-key-goes-here"


def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0125",
        model = "gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        response = chat_with_gpt(user_input)
        print("Bot:", response)
