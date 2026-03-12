# DocuMind Backend

DocuMind Backend is an AI-powered document intelligence platform built with FastAPI and LangChain. It enables seamless ingestion, processing, and querying of various document formats including PDFs, DOCX, TXT, MD, and HTML files, as well as web URLs.

The system leverages advanced retrieval-augmented generation (RAG) techniques to transform uploaded documents into searchable knowledge bases. Key features include:

- **Document Ingestion**: Automatic loading and parsing of multiple file types with metadata enrichment.
- **Intelligent Splitting**: Content chunking optimized for context windows and semantic coherence.
- **Vector Storage**: ChromaDB integration for efficient similarity search and retrieval.
- **AI-Powered Queries**: OLLAMA models for natural language question-answering over document collections.
- **Re-ranking**: Cohere integration for improved result relevance.

Designed for developers and organizations, DocuMind Backend provides RESTful APIs for document management, making it easy to integrate into existing workflows or build custom document intelligence applications.
