from generation.llm import get_llm
from generation.prompt import get_rag_prompt


def create_rag_chain():

    llm = get_llm()
    prompt = get_rag_prompt()

    chain = prompt | llm

    return chain