import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def get_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    return llm