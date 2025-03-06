from typing import List, Dict, Any, Optional
import os
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate

# Define persistent directory for Chroma
CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Define the state object
class RAGState:
    """State for the RAG application."""
    
    def __init__(
        self,
        query: str = "",
        documents: List[Document] = None,
        retrieved_documents: List[Document] = None,
        context: str = "",
        response: str = "",
        collection_id: str = None,
    ):
        self.query = query
        self.documents = documents or []
        self.retrieved_documents = retrieved_documents or []
        self.context = context
        self.response = response
        self.collection_id = collection_id

# Helper function to initialize or access Chroma DB
def get_vector_store(collection_id: str, embeddings=None):
    """Initialize or access the Chroma vector store."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=collection_id
    )

# Define retrieval function
def retrieve(state: RAGState) -> RAGState:
    """Retrieve relevant documents based on the query."""
    # If we have documents in memory, use them first
    if state.documents:
        # Create a temporary vector store
        embeddings = OpenAIEmbeddings()
        collection_id = f"temp_{uuid.uuid4().hex[:8]}"
        Chroma.from_documents(
            documents=state.documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            collection_name=collection_id
        )
        state.collection_id = collection_id
    
    # If we have a collection ID, use that for retrieval
    if state.collection_id:
        # Use the existing Chroma collection
        embeddings = OpenAIEmbeddings()
        vector_store = get_vector_store(state.collection_id, embeddings)
        
        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(state.query, k=3)
        
        # Update state
        state.retrieved_documents = retrieved_docs
    
    return state

# Define generation function
def generate_context(state: RAGState) -> RAGState:
    """Generate context from retrieved documents."""
    if not state.retrieved_documents:
        return state
    
    # Format documents into context string
    context_str = "\n\n".join([doc.page_content for doc in state.retrieved_documents])
    
    # Update state
    state.context = context_str
    return state

# Define response generation function
def generate_response(state: RAGState) -> RAGState:
    """Generate a response based on the query and context."""
    if not state.context:
        state.response = "No relevant information found."
        return state
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Use the following retrieved context to answer the user's question. 
    If you don't know the answer, just say that you don't know.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """)
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Generate response
    chain = prompt | llm
    response = chain.invoke({"context": state.context, "query": state.query})
    
    # Update state
    state.response = response.content
    return state

# Create the graph
def create_rag_graph():
    """Create a LangGraph for RAG."""
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_context", generate_context)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("retrieve", "generate_context")
    workflow.add_edge("generate_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Compile
    return workflow.compile()

# Document loading and processing
def load_and_process_documents(text_files: List[str], collection_name: str = None) -> str:
    """Load, process, and store documents in Chroma DB.
    
    Args:
        text_files: List of paths to text files
        collection_name: Optional name for the collection
        
    Returns:
        str: Collection ID for the stored documents
    """
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    for file_path in text_files:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs = splitter.create_documents([text], metadatas=[{"source": file_path}])
                documents.extend(docs)
    
    if not documents:
        return None
        
    # Generate a unique collection ID if not provided
    collection_id = collection_name or f"collection_{uuid.uuid4().hex[:8]}"
    
    # Store documents in Chroma
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=collection_id
    )
    
    return collection_id

def batch_process_documents(directory_path: str, glob_pattern: str = "*.txt", collection_name: str = None) -> str:
    """Process all documents in a directory matching the glob pattern.
    
    Args:
        directory_path: Path to directory containing documents
        glob_pattern: Pattern to match files (default: "*.txt")
        collection_name: Optional name for the collection
        
    Returns:
        str: Collection ID for the stored documents
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
        
    # Use DirectoryLoader for batch processing
    loader = DirectoryLoader(directory_path, glob=glob_pattern)
    documents = loader.load()
    
    if not documents:
        return None
        
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)
    
    # Generate a unique collection ID if not provided
    collection_id = collection_name or f"batch_{uuid.uuid4().hex[:8]}"
    
    # Store in Chroma
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=collection_id
    )
    
    return collection_id

def list_collections() -> List[str]:
    """List all available document collections in the Chroma DB."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    return [col.name for col in client.list_collections()]

# Main function to run the RAG application
def run_rag(query: str, collection_id: str = None, text_files: List[str] = None) -> str:
    """Run the RAG application.
    
    Args:
        query: User query
        collection_id: ID of existing collection to query
        text_files: Optional list of new text files to process
        
    Returns:
        str: Response to the query
    """
    # Process new documents if provided
    if text_files:
        collection_id = load_and_process_documents(text_files)
        
    if not collection_id:
        return "No documents available to search."
        
    # Create initial state
    state = RAGState(query=query, collection_id=collection_id)
    
    # Create and run graph
    graph = create_rag_graph()
    result = graph.invoke(state)
    
    return result.response

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("""
Usage: 
  Search an existing collection:
    python main.py search 'collection_id' 'Your question'
    
  Create a collection from files:
    python main.py index 'collection_name' file1.txt file2.txt ...
    
  Create a collection from a directory:
    python main.py batch 'collection_name' '/path/to/directory' '*.txt'
    
  List all collections:
    python main.py list
""")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "search" and len(sys.argv) >= 4:
        collection_id = sys.argv[2]
        query = sys.argv[3]
        response = run_rag(query=query, collection_id=collection_id)
        print("\nResponse:")
        print(response)
    
    elif command == "index" and len(sys.argv) >= 4:
        collection_name = sys.argv[2]
        files = sys.argv[3:]
        collection_id = load_and_process_documents(files, collection_name)
        print(f"Created collection: {collection_id}")
    
    elif command == "batch" and len(sys.argv) >= 5:
        collection_name = sys.argv[2]
        directory = sys.argv[3]
        pattern = sys.argv[4]
        collection_id = batch_process_documents(directory, pattern, collection_name)
        print(f"Created collection: {collection_id}")
    
    elif command == "list":
        collections = list_collections()
        if collections:
            print("Available collections:")
            for collection in collections:
                print(f"  - {collection}")
        else:
            print("No collections found.")
    
    else:
        print("Invalid command or arguments.")
        sys.exit(1)