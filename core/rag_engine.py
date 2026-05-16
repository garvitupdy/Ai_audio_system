from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .vector_store import build_vector_store, load_vector_store, get_retriever
import os

def get_llm():
    """Initialize LLM"""
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.3,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

def build_rag_chain(transcript: str, transcript_id: int = None):
    """
    Build RAG chain for question answering
    
    Args:
        transcript: The transcript text
        transcript_id: ID for persisting the vector store
    
    Returns:
        RAG chain
    """
    print("🔧 Building RAG chain...")
    
  
    vector_store = build_vector_store(transcript, transcript_id)
    retriever = get_retriever(vector_store, k=4)
    

    template = """You are an AI assistant helping to answer questions about a meeting transcript.
Use the following context to answer the question. If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()
    
  
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG chain built successfully")
    return rag_chain


def load_rag_chain(transcript_id: int):
    """
    Load existing RAG chain for a transcript
    
    Args:
        transcript_id: ID of the transcript
    
    Returns:
        RAG chain
    """
    print(f"🔧 Loading RAG chain for transcript {transcript_id}...")
    
    
    vector_store = load_vector_store(transcript_id)
    retriever = get_retriever(vector_store, k=4)
   
    template = """You are an AI assistant helping to answer questions about a meeting transcript.
Use the following context to answer the question. If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()
    
    
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG chain loaded successfully")
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    """
    Ask a question using the RAG chain
    
    Args:
        rag_chain: The RAG chain
        question: The question to ask
    
    Returns:
        Answer string
    """
    return rag_chain.invoke(question)