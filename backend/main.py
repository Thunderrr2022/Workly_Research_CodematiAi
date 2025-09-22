"""
Deep Researcher Agent - Main FastAPI Application
A multi-agent research system with full chain-of-thought transparency.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import tempfile
import shutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_researcher.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from agents.retriever_agent import RetrieverAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.query_splitter import QuerySplitter
from agents.reasoning_logger import ReasoningLogger
from agents.gap_detector import KnowledgeGapDetector
from agents.report_exporter import ReportExporter
from agents.document_research_agent import DocumentResearchAgent
from config import config

# Initialize FastAPI app
app = FastAPI(
    title="Deep Researcher Agent",
    description="Multi-agent research system with full chain-of-thought transparency",
    version="1.0.0"
)

# Configure CORS for frontend integration
allowed_origins = config.ALLOWED_ORIGINS
if config.DEBUG:
    # In dev, allow all localhost ports (vite may pick 5173/5174/5175)
    allowed_origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
retriever_agent = RetrieverAgent()
synthesizer_agent = SynthesizerAgent()
query_splitter = QuerySplitter()
reasoning_logger = ReasoningLogger()
gap_detector = KnowledgeGapDetector()
report_exporter = ReportExporter()
document_research_agent = DocumentResearchAgent(
    retriever_agent=retriever_agent,
    synthesizer_agent=synthesizer_agent,
    reasoning_logger=reasoning_logger,
    gap_detector=gap_detector
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    documents: Optional[List[str]] = []  # Document IDs or paths

class QueryResponse(BaseModel):
    answer: str
    reasoning_trace: List[str]
    gaps: List[str]

class DocumentResearchRequest(BaseModel):
    doc_id: str
    query: Optional[str] = None

class DocumentResearchResponse(BaseModel):
    document_info: Dict[str, Any]
    research_questions: List[str]
    insights: List[Dict[str, Any]]
    reasoning_trace: List[str]
    gaps: List[str]
    retrieved_sections: int
    session_id: str

class ExportRequest(BaseModel):
    query: str
    answer: str
    reasoning_trace: List[str]
    gaps: List[str]
    documents: List[str]
    format: str  # 'pdf' or 'markdown'
    research_type: str = "query"  # 'query' or 'document'

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Deep Researcher Agent API is running", "status": "healthy"}

@app.post("/upload", response_model=dict)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents for research.
    Returns document IDs that can be used in queries.
    """
    try:
        uploaded_docs = []
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith(('.txt', '.pdf', '.docx')):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            # Process document with retriever agent
            try:
                doc_id = retriever_agent.add_document(tmp_path, file.filename)
            except Exception as e:
                # Map extraction issues to a clean, user-friendly error
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from this document. Try a text-based PDF/DOCX/TXT or enable OCR for scanned PDFs."
                )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            uploaded_docs.append({
                "id": doc_id,
                "name": file.filename,
                "size": len(content),
                "type": file.content_type
            })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_docs)} documents",
            "documents": uploaded_docs
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed due to an unexpected server error.")

@app.post("/research/document/upload")
async def research_document_upload(files: List[UploadFile] = File(...)):
    """
    Upload and research documents directly.
    Accepts file uploads (TXT, PDF, DOCX) and performs comprehensive document research.
    Returns structured JSON with answer, reasoning_trace, and gaps.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Process only the first file for document research
        file = files[0]
        logger.info(f"Processing document: {file.filename}")
        
        # Validate file type
        valid_extensions = ['.txt', '.pdf', '.docx']
        file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        if file_extension not in valid_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(valid_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from {file.filename}")
        
        # Generate unique filename with UUID
        import uuid
        unique_id = str(uuid.uuid4())
        safe_filename = f"{unique_id}_{file.filename}"
        
        # Save file locally with unique ID
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, safe_filename)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved file to: {file_path}")
        
        try:
            # Process the document and generate embeddings immediately
            logger.info(f"Starting document processing for: {file.filename}")
            try:
                doc_id = retriever_agent.add_document(file_path, file.filename)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from this document. Try a text-based PDF/DOCX/TXT or enable OCR for scanned PDFs."
                )
            logger.info(f"Document processed with ID: {doc_id}")
            
            if not doc_id:
                logger.error(f"Failed to process document: {file.filename}")
                raise HTTPException(status_code=500, detail="Failed to process document")
            
            # Verify document is immediately searchable
            all_docs = retriever_agent.list_documents()
            doc_exists = any(doc['id'] == doc_id for doc in all_docs)
            logger.info(f"Document {doc_id} indexed and searchable: {doc_exists}")
            logger.info(f"Total documents in index: {len(all_docs)}")
            
            if not doc_exists:
                logger.error(f"Document {doc_id} not found after processing")
                raise HTTPException(status_code=500, detail=f"Document {doc_id} not found after processing")
            
            # Initialize a fresh per-session store for document research (clears old state)
            try:
                # Build a session-scoped index strictly for this doc
                retriever_agent.document_session_doc_id = doc_id
                # Use stored per-doc embeddings
                emb = retriever_agent.doc_embeddings.get(doc_id)
                chunks = retriever_agent.doc_chunks.get(doc_id, [])
                if emb is not None and emb.shape[0] > 0:
                    sess_index = faiss.IndexFlatIP(emb.shape[1])
                    sess_index.add(emb)
                    retriever_agent.document_session_index = sess_index
                    retriever_agent.document_session_chunks = chunks
                logger.info(f"[Session] Initialized document session for doc_id={doc_id} with {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"[Session] Failed to initialize session-scoped index: {e}")

            # Perform document research with improved synthesis
            logger.info(f"Starting document research for: {doc_id}")
            logger.info(f"[Research Scope] Running document research on doc_id: {doc_id}")
            result = document_research_agent.research_document(doc_id)
            logger.info(f"Document research completed for: {doc_id}")
            
            if "error" in result:
                logger.error(f"Document research error: {result['error']}")
                raise HTTPException(status_code=400, detail=result["error"])
            
            # Log research results
            logger.info(f"Research results - Answer length: {len(result.get('answer', ''))}, Reasoning steps: {len(result.get('reasoning_trace', []))}, Gaps: {len(result.get('gaps', []))}")
            
            # Return simplified structured JSON as requested
            response_data = {
                "answer": result.get("answer", "No analysis available"),
                "reasoning_trace": result.get("reasoning_trace", []),
                "gaps": result.get("gaps", []),
                "document_info": result.get("document_info", {}),
                "session_id": result.get("session_id", "")
            }
            
            logger.info(f"Returning response for document research: {file.filename}")
            return response_data
            
        finally:
            # Reset session store after research completes
            retriever_agent.document_session_doc_id = None
            retriever_agent.document_session_index = None
            retriever_agent.document_session_chunks = []
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document research upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/research/document")
async def research_document(request: DocumentResearchRequest):
    """
    Perform comprehensive research on a specific document by ID.
    This is independent of query research and analyzes the document directly.
    """
    try:
        # If a custom query is provided, constrain retrieval to this doc but tailor synthesis to the query
        if request.query:
            reasoning_logger.start_session()
            reasoning_logger.log_step("Document Query", f"Tailored query on document {request.doc_id}: '{request.query}'")
            docs = retriever_agent.retrieve_documents(request.query, top_k=8, doc_id=request.doc_id)
            if not docs:
                answer = f"No relevant information found in this document for: '{request.query}'. Try uploading additional sources or refining your query."
            else:
                answer = synthesizer_agent.synthesize(request.query, docs, [request.query])
            trace = reasoning_logger.get_trace()
            gaps = gap_detector.detect_gaps(request.query, docs, [request.query])
            sources = []
            if docs:
                by_src = {}
                for d in docs:
                    key = (d.get('source','Unknown'), d.get('doc_id',''))
                    by_src.setdefault(key, set()).add(d.get('page'))
                for (src, did), pages in by_src.items():
                    page_list = sorted([p for p in pages if p is not None])
                    sources.append({
                        "filename": src,
                        "doc_id": did,
                        "pages_used": page_list
                    })
            return {
                "answer": answer,
                "document_info": next((d for d in retriever_agent.list_documents() if d['id']==request.doc_id), {}),
                "research_questions": [request.query],
                "insights": [],
                "reasoning_trace": trace,
                "gaps": gaps,
                "retrieved_sections": len(docs),
                "session_id": "",
                "sources": sources
            }
        
        # Otherwise run the full document research workflow
        result = document_research_agent.research_document(request.doc_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return DocumentResearchResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error researching document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main research endpoint that processes queries through the multi-agent system.
    Returns synthesized answer with reasoning trace and knowledge gaps.
    """
    try:
        # Step 1: Log initial query
        reasoning_logger.log_step("Query received", f"Processing query: '{request.query}'")
        
        # Step 2: Split query into sub-questions
        sub_queries = query_splitter.split_query(request.query)
        reasoning_logger.log_step(
            "Query decomposition", 
            f"Broke down query into {len(sub_queries)} sub-questions: {', '.join(sub_queries)}"
        )
        
        # Step 3: Retrieve relevant documents for each sub-query
        retrieved_docs = []
        for sub_query in sub_queries:
            docs = retriever_agent.retrieve_documents(sub_query, top_k=5)
            retrieved_docs.extend(docs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc['content'] not in seen:
                seen.add(doc['content'])
                unique_docs.append(doc)
        
        reasoning_logger.log_step(
            "Document retrieval", 
            f"Retrieved {len(unique_docs)} relevant document sections from {len(set(doc['source'] for doc in unique_docs))} sources"
        )
        
        # Step 4: Synthesize information
        if unique_docs:
            synthesized_answer = synthesizer_agent.synthesize(
                request.query, 
                unique_docs, 
                sub_queries
            )
        else:
            synthesized_answer = "No relevant documents found for this query. Please upload relevant documents or try a different query."
        
        reasoning_logger.log_step(
            "Information synthesis", 
            f"Synthesized comprehensive answer from {len(unique_docs)} document sections"
        )
        
        # Step 5: Detect knowledge gaps
        gaps = gap_detector.detect_gaps(request.query, unique_docs, sub_queries)
        reasoning_logger.log_step(
            "Gap analysis", 
            f"Identified {len(gaps)} knowledge gaps in the research"
        )
        
        # Get reasoning trace
        reasoning_trace = reasoning_logger.get_trace()
        
        return QueryResponse(
            answer=synthesized_answer,
            reasoning_trace=reasoning_trace,
            gaps=gaps
        )
    
    except Exception as e:
        reasoning_logger.log_step("Error", f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/export")
async def export_report(request: ExportRequest):
    """
    Export research report in PDF or Markdown format.
    Supports both query research and document research.
    """
    try:
        # Generate report
        if request.format.lower() == 'pdf':
            file_path = report_exporter.export_pdf(
                query=request.query,
                answer=request.answer,
                reasoning_trace=request.reasoning_trace,
                gaps=request.gaps,
                documents=request.documents,
                research_type=request.research_type
            )
            media_type = "application/pdf"
        elif request.format.lower() == 'markdown':
            file_path = report_exporter.export_markdown(
                query=request.query,
                answer=request.answer,
                reasoning_trace=request.reasoning_trace,
                gaps=request.gaps,
                documents=request.documents,
                research_type=request.research_type
            )
            media_type = "text/markdown"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'pdf' or 'markdown'")
        
        # Return file
        report_type = "document_research" if request.research_type == "document" else "query_research"
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format.lower()}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting report: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = retriever_agent.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document"""
    try:
        success = retriever_agent.delete_document(doc_id)
        if success:
            return {"message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/vector_store", exist_ok=True)
    os.makedirs("data/reports", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
