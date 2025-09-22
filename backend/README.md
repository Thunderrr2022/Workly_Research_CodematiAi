# Deep Researcher Agent - Backend

A fully functional multi-agent research system with complete chain-of-thought transparency.

## Features

- **Multi-Agent Architecture**: RetrieverAgent, SynthesizerAgent, QuerySplitter, ReasoningLogger, KnowledgeGapDetector, ReportExporter
- **Document Processing**: Supports TXT, PDF, DOCX files with intelligent text extraction and chunking
- **Vector Search**: FAISS-based semantic search with sentence transformers
- **LLM Integration**: OpenAI API integration for intelligent query decomposition and synthesis
- **Full Transparency**: Complete reasoning trace logging for every research step
- **Gap Detection**: Automatic identification of knowledge gaps and missing information
- **Report Generation**: Dynamic PDF and Markdown report export
- **RESTful API**: FastAPI-based endpoints for seamless frontend integration

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the backend directory:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Alternative: HuggingFace Configuration
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium

# Vector Store Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=./data/vector_store
MAX_DOCUMENT_SIZE=10000000

# Document Processing
SUPPORTED_FORMATS=txt,pdf,docx
MAX_DOCUMENTS_PER_QUERY=50

# Report Generation
REPORT_OUTPUT_DIR=./data/reports
DEFAULT_REPORT_FORMAT=pdf

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

### 3. Run the Backend

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 4. API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Core Endpoints

- `POST /query` - Process research queries
- `POST /export` - Export research reports
- `POST /upload` - Upload documents for analysis
- `GET /documents` - List uploaded documents
- `DELETE /documents/{doc_id}` - Delete specific document

### Health Check

- `GET /` - Health check endpoint

## Architecture

### Agents

1. **RetrieverAgent**: Document processing, embedding generation, FAISS vector search
2. **SynthesizerAgent**: Information synthesis using LLM or fallback methods
3. **QuerySplitter**: Intelligent query decomposition into sub-questions
4. **ReasoningLogger**: Complete chain-of-thought transparency logging
5. **KnowledgeGapDetector**: Missing information identification
6. **ReportExporter**: Dynamic PDF/Markdown report generation

### Data Flow

1. **Query Input** → QuerySplitter (decompose into sub-questions)
2. **Sub-queries** → RetrieverAgent (semantic search in documents)
3. **Retrieved Documents** → SynthesizerAgent (combine into coherent answer)
4. **All Data** → KnowledgeGapDetector (identify missing information)
5. **Complete Process** → ReasoningLogger (log all steps)
6. **Final Result** → ReportExporter (generate reports)

## Configuration

All configuration is handled through environment variables or the `config.py` file. Key settings:

- **LLM Provider**: OpenAI (recommended) or HuggingFace
- **Embedding Model**: Sentence transformers model for vector embeddings
- **Vector Store**: FAISS index for fast similarity search
- **Document Formats**: TXT, PDF, DOCX support
- **Report Formats**: PDF and Markdown export

## Error Handling

The system includes comprehensive error handling:

- Graceful fallbacks when LLM services are unavailable
- Document processing error recovery
- Vector store corruption handling
- API error responses with detailed messages

## Performance

- **Vector Search**: Sub-second retrieval from thousands of documents
- **Document Processing**: Efficient chunking and embedding generation
- **LLM Integration**: Optimized prompts for faster responses
- **Report Generation**: Fast PDF/Markdown export

## Security

- Input validation and sanitization
- File type restrictions
- Size limits on uploads
- CORS configuration for frontend integration

## Development

### Adding New Document Types

1. Extend `RetrieverAgent._extract_text()` method
2. Add file extension to `SUPPORTED_FORMATS` config
3. Update frontend file type validation

### Customizing LLM Prompts

Modify prompt templates in:
- `SynthesizerAgent._create_synthesis_prompt()`
- `QuerySplitter._split_with_llm()`
- `KnowledgeGapDetector._detect_gaps_with_llm()`

### Adding New Report Formats

1. Extend `ReportExporter` class
2. Add new export method
3. Update API endpoint to handle new format

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check API key and rate limits
2. **Document Processing Failures**: Verify file formats and sizes
3. **Vector Store Issues**: Delete `data/vector_store` to reset
4. **Memory Issues**: Reduce `MAX_DOCUMENT_SIZE` or chunk size

### Logs

Check application logs for detailed error information:
```bash
tail -f logs/app.log
```

## License

This project is part of the Deep Researcher Agent system.


