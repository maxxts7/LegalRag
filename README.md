# LangGraph RAG Application

A simple Retrieval Augmented Generation (RAG) application built with LangGraph and LangChain.

## Features

- Query-based document retrieval
- Context generation from relevant documents
- LLM-powered response generation using OpenAI models
- Simple command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/langgraph-rag.git
cd langgraph-rag

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt
```

## Usage

First, set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Run the application:

```bash
python app/main.py "Your question here" app/sample_data.txt additional_file.txt
```

Example:

```bash
python app/main.py "What is LangGraph used for?" app/sample_data.txt
```

## How It Works

1. **Document Processing**: Text files are loaded and split into chunks
2. **Retrieval**: User query is used to find relevant document chunks
3. **Context Generation**: Retrieved chunks are combined into context
4. **Response Generation**: LLM generates an answer based on context and query

## Project Structure

- `app/main.py`: Main application code with LangGraph implementation
- `app/requirements.txt`: Dependencies
- `app/sample_data.txt`: Sample data for testing

## Requirements

- Python 3.9+
- OpenAI API key
- Dependencies listed in requirements.txt