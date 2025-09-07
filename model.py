import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")


def call_llm(prompt):
    client = OpenAI(api_key=api_key, base_url=base_url)
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


if __name__ == '__main__':
    print(call_llm("who are you?"))
