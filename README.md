# Deep Researcher Agent

A fully functional multi-agent research system with complete chain-of-thought transparency, built with React frontend and Python backend.

## 🌟 Features

### Multi-Agent Architecture
- **RetrieverAgent**: Dynamic document processing (TXT, PDF, DOCX) with FAISS vector search
- **SynthesizerAgent**: Intelligent information synthesis using OpenAI or fallback methods
- **QuerySplitter**: Dynamic query decomposition into focused sub-questions
- **ReasoningLogger**: Complete chain-of-thought transparency for every research step
- **KnowledgeGapDetector**: Automatic identification of missing information
- **ReportExporter**: Dynamic PDF and Markdown report generation

### Full Integration
- **Dynamic Backend**: Fully functional Python FastAPI backend with no static responses
- **Real-time Processing**: Live document upload, processing, and analysis
- **End-to-End Workflow**: From query input to comprehensive report export
- **Error Handling**: Graceful fallbacks and comprehensive error management

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- OpenAI API key (recommended) or HuggingFace API key

### 1. Start the Backend

```bash
# Make the startup script executable (if not already)
chmod +x start_backend.sh

# Start the backend server
./start_backend.sh
```

The backend will be available at `http://localhost:8000`

### 2. Start the Frontend

```bash
# Install frontend dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### 3. Configure API Keys

Edit `backend/.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 📖 Usage

### 1. Upload Documents (Optional)
- Drag and drop or click to upload TXT, PDF, or DOCX files
- Documents are automatically processed and indexed for search

### 2. Enter Research Query
- Type your research question in the input field
- The system will automatically decompose complex queries into sub-questions

### 3. View Results
- **Reasoning Trace**: See every step of the research process
- **Synthesized Answer**: Comprehensive answer based on retrieved information
- **Knowledge Gaps**: Identified areas where information is missing

### 4. Export Reports
- Click "Export PDF" or "Export MD" to download comprehensive reports
- Reports include reasoning trace, citations, and knowledge gaps

## 🏗️ Architecture

### Frontend (React + TypeScript)
```
src/
├── App.tsx                 # Main application component
├── components/
│   ├── QueryInput.tsx      # Query input interface
│   ├── DocumentUpload.tsx  # Document upload component
│   ├── ReasoningTrace.tsx  # Reasoning process display
│   ├── GapsDisplay.tsx     # Knowledge gaps display
│   ├── AnswerDisplay.tsx   # Final answer display
│   └── LoadingState.tsx    # Loading animations
```

### Backend (Python + FastAPI)
```
backend/
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── agents/
│   ├── retriever_agent.py     # Document processing & search
│   ├── synthesizer_agent.py   # Information synthesis
│   ├── query_splitter.py      # Query decomposition
│   ├── reasoning_logger.py    # Process transparency
│   ├── gap_detector.py        # Knowledge gap detection
│   └── report_exporter.py     # Report generation
└── data/                   # Data storage
    ├── documents/          # Uploaded documents
    ├── vector_store/       # FAISS index
    └── reports/           # Generated reports
```

## 🔧 API Endpoints

### Core Endpoints
- `POST /query` - Process research queries
- `POST /export` - Export research reports (PDF/Markdown)
- `POST /upload` - Upload documents for analysis
- `GET /documents` - List uploaded documents
- `DELETE /documents/{doc_id}` - Delete specific document

### Health Check
- `GET /` - Health check endpoint

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)

## 🎯 Example Queries

Try these example queries to see the system in action:

1. **"How is artificial intelligence transforming healthcare?"**
2. **"What are the environmental impacts of renewable energy adoption?"**
3. **"Analyze the economic effects of remote work on urban planning"**

## 🔍 How It Works

### 1. Query Processing
```
User Query → QuerySplitter → Sub-questions
```

### 2. Document Retrieval
```
Sub-questions → RetrieverAgent → Relevant Documents
```

### 3. Information Synthesis
```
Documents + Sub-questions → SynthesizerAgent → Comprehensive Answer
```

### 4. Gap Detection
```
Answer + Documents → KnowledgeGapDetector → Missing Information
```

### 5. Report Generation
```
All Data → ReportExporter → PDF/Markdown Report
```

## 🛠️ Configuration

### Environment Variables
```env
# LLM Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Vector Store
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=./data/vector_store

# Document Processing
SUPPORTED_FORMATS=txt,pdf,docx
MAX_DOCUMENT_SIZE=10000000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## 📊 Performance

- **Document Processing**: Handles files up to 10MB
- **Vector Search**: Sub-second retrieval from thousands of documents
- **LLM Integration**: Optimized prompts for faster responses
- **Report Generation**: Fast PDF/Markdown export

## 🔒 Security

- Input validation and sanitization
- File type restrictions
- Size limits on uploads
- CORS configuration for frontend integration

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure backend is running on `http://localhost:8000`
   - Check if all dependencies are installed
   - Verify Python version (3.10+ required)

2. **OpenAI API Errors**
   - Verify API key is correct
   - Check rate limits and billing
   - System will fallback to rule-based methods

3. **Document Upload Issues**
   - Ensure files are TXT, PDF, or DOCX format
   - Check file size (max 10MB)
   - Verify backend is running

4. **Memory Issues**
   - Reduce `MAX_DOCUMENT_SIZE` in config
   - Clear vector store: `rm -rf backend/data/vector_store/*`

### Logs
Check backend logs for detailed error information:
```bash
tail -f backend/logs/app.log
```

## 🚀 Deployment

### Development
```bash
# Backend
cd backend && python main.py

# Frontend
npm run dev
```

### Production
```bash
# Build frontend
npm run build

# Serve with nginx or similar
# Backend can be deployed with gunicorn
```

## 📝 License

This project is part of the Deep Researcher Agent system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `http://localhost:8000/docs`
3. Check backend logs for detailed error information

---

**Deep Researcher Agent** - Multi-agent research system with full chain-of-thought transparency.


