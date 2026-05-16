import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    """Get HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": 'cpu'}
    )

def build_vector_store(transcript: str, transcript_id: int = None) -> Chroma:
    """
    Build vector store for a specific transcript
    
    Args:
        transcript: The text to vectorize
        transcript_id: Unique ID for this transcript (used for persistence)
    
    Returns:
        Chroma vector store
    """
    print(f"Building vector store for transcript {transcript_id}...")

    
    collection_name = f"transcript_{transcript_id}" if transcript_id else "temp_transcript"
    persist_dir = f"database/vectors/chroma_{transcript_id}" if transcript_id else "database/vectors/temp_chroma"
    
    
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    
    docs = [
        Document(
            page_content=chunk, 
            metadata={
                'chunk_index': i,
                'transcript_id': transcript_id,
                'total_chunks': len(chunks)
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    print(f"Created {len(docs)} chunks from transcript")

    # Create vector store
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )

    print(f"✅ Vector store created and persisted to {persist_dir}")
    return vector_store


def load_vector_store(transcript_id: int) -> Chroma:
    """
    Load existing vector store for a specific transcript
    
    Args:
        transcript_id: ID of the transcript to load
    
    Returns:
        Chroma vector store
    """
    collection_name = f"transcript_{transcript_id}"
    persist_dir = f"database/vectors/chroma_{transcript_id}"
    
  
    if not Path(persist_dir).exists():
        raise FileNotFoundError(f"No vector store found for transcript {transcript_id}")
    
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"✅ Loaded vector store for transcript {transcript_id}")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Get retriever from vector store
    
    Args:
        vector_store: Chroma vector store
        k: Number of documents to retrieve
    
    Returns:
        Retriever object
    """
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )


def delete_vector_store(transcript_id: int):
    """
    Delete vector store for a specific transcript
    
    Args:
        transcript_id: ID of the transcript to delete
    """
    import shutil
    
    persist_dir = Path(f"database/vectors/chroma_{transcript_id}")
    
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"🗑️ Deleted vector store for transcript {transcript_id}")
    else:
        print(f"⚠️ No vector store found for transcript {transcript_id}")


def cleanup_old_vectors(keep_ids: list):
    """
    Clean up vector stores that are no longer in the database
    
    Args:
        keep_ids: List of transcript IDs to keep
    """
    import shutil
    
    vectors_dir = Path("database/vectors")
    if not vectors_dir.exists():
        return
    

    for item in vectors_dir.iterdir():
        if item.is_dir() and item.name.startswith("chroma_"):
            try:
            
                transcript_id = int(item.name.replace("chroma_", ""))
                
                
                if transcript_id not in keep_ids:
                    shutil.rmtree(item)
                    print(f"🗑️ Cleaned up orphaned vector store: {item.name}")
            except ValueError:
               
                continue
