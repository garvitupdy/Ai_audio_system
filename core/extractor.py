
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.summarizer import split_transcript
import os 
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatGroq(model = "llama-3.3-70b-versatile",groq_api_key = os.getenv("GROQ_API_KEY"),temperature=0.2)




def build_chain(system_prompt : str):

    


    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    llm = get_llm()
    chunks_ai = split_transcript(transcript)

    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    chunk_action = [chain.invoke({'text': chunk}) for chunk in chunks_ai]

    combined_action = "\n\n".join(chunk_action)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are a meeting expert who extract all action items from meetings. Combine these partial action items from different chunks of meeting "
            "into one final professional meeting action items in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined_action)


    


def extract_key_decisions(transcript: str) -> str:
    llm = get_llm()
    chunks_kd = split_transcript(transcript)
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    
    chunk_decisions = [chain.invoke({'text': chunk}) for chunk in chunks_kd]

    combined_decisions = "\n\n".join(chunk_decisions)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are a meeting expert who extract all key decisions from meetings. Combine these partial key decisions from different chunks of meeting "
            "into one final professional meeting key decisions in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined_decisions)

def extract_questions(transcript: str) -> str:
    chunks_eq = split_transcript(transcript)
    llm = get_llm()
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    chunk_questions = [chain.invoke({'text': chunk}) for chunk in chunks_eq]

    combined_questions = "\n\n".join(chunk_questions)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are a meeting expert who extract all key decisions from meetings. Combine these partial key decisions from different chunks of meeting "
            "into one final professional meeting key decisions in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined_questions)