# Vector Database RAG System

A Retrieval-Augmented Generation (RAG) system using Pinecone vector database and Google Vertex AI for intelligent document retrieval and question answering.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI technique that combines information retrieval with text generation. Instead of relying solely on pre-trained knowledge, RAG systems:

1. **Retrieve** relevant documents from a knowledge base using vector similarity search
2. **Augment** the user's query with retrieved context
3. **Generate** accurate, contextual responses using a language model

This approach reduces hallucinations and provides more accurate, up-to-date answers based on your specific documents.

## Project Description

This project implements a complete RAG pipeline that:
- Stores document embeddings in Pinecone vector database
- Retrieves relevant context based on user queries
- Generates responses using Google's Gemini model
- Provides both standard and custom-formatted answers

## Features

- Document retrieval from Pinecone vector store
- Text embeddings using Vertex AI
- Question answering with Gemini LLM
- Custom RAG chain implementation
- Real-time document ingestion and querying

## Setup

1. Install dependencies:
```bash
pipenv install
```

2. Create `.env` file with:
```
INDEX_NAME=your_pinecone_index_name
```

3. Configure Google Cloud credentials for Vertex AI access

## Usage

```bash
pipenv run python main.py
```

The system will:
1. Query the vector database for relevant documents
2. Generate a concise answer using retrieved context
3. Display both detailed and custom formatted responses

## File Structure

- **`main.py`** - Main application that runs the RAG system, retrieves documents, and generates answers
- **`ingestion.py`** - Script for processing and uploading documents to Pinecone vector database
- **`mediumblog1.txt`** - Sample document containing information about vector databases
- **`.env`** - Environment variables (INDEX_NAME for Pinecone)
- **`Pipfile`** - Pipenv dependency management file

## Dependencies

- **langchain** - Framework for building LLM applications
- **langchain-google-vertexai** - Google Vertex AI integration for embeddings and LLM
- **langchain-pinecone** - Pinecone vector database integration
- **python-dotenv** - Environment variable management