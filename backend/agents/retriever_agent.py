"""
RetrieverAgent - Document processing and retrieval system
Handles document ingestion, embedding generation, and semantic search using FAISS.
"""

import os
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import re
import io

from config import config

logger = logging.getLogger(__name__)

class RetrieverAgent:
    """
    Multi-modal document retriever that processes various document formats,
    generates embeddings, and provides semantic search capabilities.
    """
    
    def __init__(self):
        """Initialize the RetrieverAgent with embedding model and vector store."""
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.vector_store_path = config.VECTOR_STORE_PATH
        self.documents = {}  # Document metadata storage
        self.index = None  # Deprecated global index (kept for backward compat, not used in retrieval)
        self.document_embeddings = []
        self.document_texts = []
        # Per-document in-memory stores for strict doc-scoped retrieval
        self.doc_embeddings: Dict[str, np.ndarray] = {}
        self.doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.doc_indices: Dict[str, faiss.IndexFlatIP] = {}
        self.current_doc_id: Optional[str] = None
        # Session-scoped store for single-document research
        self.document_session_doc_id: Optional[str] = None
        self.document_session_index: Optional[faiss.IndexFlatIP] = None
        self.document_session_chunks: List[Dict[str, Any]] = []
        
        # Initialize or load existing vector store
        self._initialize_vector_store()
        
        logger.info("RetrieverAgent initialized successfully")
    
    def _initialize_vector_store(self):
        """Initialize or load existing FAISS vector store."""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        metadata_path = os.path.join(self.vector_store_path, "metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load existing index
                self.index = faiss.read_index(index_path)
                self._load_metadata(metadata_path)
                logger.info(f"Loaded existing vector store with {len(self.documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}. Creating new one.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Get embedding dimension from the model
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        logger.info(f"Created new FAISS index with dimension {embedding_dim}")
    
    def _load_metadata(self, metadata_path: str):
        """Load document metadata from file."""
        import json
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get('documents', {})
                self.document_texts = data.get('texts', [])
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.documents = {}
            self.document_texts = []
    
    def _save_metadata(self):
        """Save document metadata to file."""
        import json
        metadata_path = os.path.join(self.vector_store_path, "metadata.json")
        try:
            data = {
                'documents': self.documents,
                'texts': self.document_texts
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _save_index(self):
        """Save FAISS index to disk."""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        try:
            faiss.write_index(self.index, index_path)
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file with OCR fallback for scanned PDFs."""
        try:
            text = ""
            
            # First try pdfplumber for better text extraction
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text from PDF using pdfplumber: {len(text)} characters")
                    return text.strip()
            except Exception as e:
                logger.warning(f"pdfplumber failed for {file_path}: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text from PDF using PyPDF2: {len(text)} characters")
                    return text.strip()
            except Exception as e:
                logger.warning(f"PyPDF2 failed for {file_path}: {e}")
            
            # OCR fallback for scanned PDFs
            try:
                logger.info(f"Attempting OCR extraction for scanned PDF: {file_path}")
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Convert page to image
                        page_image = page.to_image(resolution=300)
                        pil_image = page_image.original
                        
                        # Extract text using OCR
                        ocr_text = pytesseract.image_to_string(pil_image, lang='eng')
                        if ocr_text and ocr_text.strip():
                            text += f"\n--- Page {page_num + 1} (OCR) ---\n{ocr_text}\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text using OCR: {len(text)} characters")
                    return text.strip()
                else:
                    raise Exception("OCR extraction produced no text")
                    
            except Exception as e:
                logger.error(f"OCR extraction failed for {file_path}: {e}")
                raise Exception(f"Could not extract text from this PDF. The file may be corrupted, password-protected, or contain only images without readable text. Error: {str(e)}")
                
        except Exception as e:
            logger.error(f"All PDF extraction methods failed for {file_path}: {e}")
            raise Exception(f"Could not extract text from this PDF. Please try another file or ensure the PDF contains readable text. Error: {str(e)}")
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text content from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            if not text.strip():
                raise Exception("No readable text found in DOCX file")
            
            logger.info(f"Successfully extracted text from DOCX: {len(text)} characters")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise Exception(f"Could not extract text from this DOCX file. The file may be corrupted or contain no readable text. Error: {str(e)}")
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text content from TXT file."""
        try:
            # Try UTF-8 first
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
            except UnicodeDecodeError:
                # Fallback to latin-1
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read().strip()
            
            if not text:
                raise Exception("TXT file is empty or contains no readable text")
            
            logger.info(f"Successfully extracted text from TXT: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            raise Exception(f"Could not extract text from this TXT file. The file may be corrupted or contain no readable text. Error: {str(e)}")
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from various document formats."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_text_from_txt(file_path)
            else:
                raise Exception(f"Unsupported file format: {file_extension}. Supported formats: PDF, DOCX, TXT")
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            raise e
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start + chunk_size // 2:  # Don't make chunks too small
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def _generate_document_id(self, file_path: str, content: str) -> str:
        """Generate a unique document ID per upload (filename + content hash + uuid)."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        filename = os.path.basename(file_path)
        return f"{filename}_{content_hash}_{uuid.uuid4().hex[:8]}"
    
    def add_document(self, file_path: str, filename: str) -> str:
        """
        Add a document to the vector store.
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            
        Returns:
            Document ID
        """
        try:
            # Extract text content
            text = self._extract_text(file_path)
            if not text:
                raise ValueError(f"No text content extracted from {filename}")
            
            # Check document size
            if len(text) > config.MAX_DOCUMENT_SIZE:
                raise ValueError(f"Document too large: {len(text)} characters (max: {config.MAX_DOCUMENT_SIZE})")
            
            # Generate document ID
            doc_id = self._generate_document_id(file_path, text)
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Generate normalized embeddings for each chunk (per-document only)
            embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True).astype('float32')
            
            # Store document metadata
            self.documents[doc_id] = {
                'id': doc_id,
                'filename': filename,
                'file_path': file_path,
                'text_length': len(text),
                'chunk_count': len(chunks),
                'added_at': str(np.datetime64('now'))
            }
            
            # Store document texts with best-effort page reference
            page_regex = re.compile(r"---\s*Page\s*(\d+)", re.IGNORECASE)
            doc_chunks: List[Dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                page_match = page_regex.search(chunk)
                page_num = int(page_match.group(1)) if page_match else None
                entry = {
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'content': chunk,
                    'filename': filename,
                    'page': page_num
                }
                self.document_texts.append(entry)
                doc_chunks.append(entry)
            # Save per-document embeddings and chunks for strict doc retrieval
            self.doc_embeddings[doc_id] = embeddings
            self.doc_chunks[doc_id] = doc_chunks
            # Build per-document FAISS index
            if embeddings.shape[0] > 0:
                doc_index = faiss.IndexFlatIP(embeddings.shape[1])
                doc_index.add(embeddings)
                self.doc_indices[doc_id] = doc_index
            
            # Save to disk
            # Global index no longer used for retrieval; keep metadata only
            self._save_metadata()
            
            logger.info(f"Successfully added document {filename} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document {filename}: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 10, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            doc_id: Optional document ID to limit search to specific document
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate normalized query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True).astype('float32')
            results: List[Dict[str, Any]] = []

            if doc_id:
                # Prefer session-scoped index if active
                if self.document_session_doc_id == doc_id and self.document_session_index is not None:
                    logger.info(f"[Doc-Scoped][Session] Searching doc_id={doc_id} (chunks={len(self.document_session_chunks)})")
                    scores, indices = self.document_session_index.search(query_embedding, min(top_k, len(self.document_session_chunks)))
                    for score, idx in zip(scores[0], indices[0]):
                        if 0 <= idx < len(self.document_session_chunks):
                            dt = self.document_session_chunks[idx]
                            # Enforce filter
                            if dt.get('doc_id') != doc_id:
                                logger.warning(f"[Doc-Scoped][Session] Skipping chunk with mismatched doc_id={dt.get('doc_id')} (expected {doc_id})")
                                continue
                            meta = self.documents.get(doc_id, {})
                            results.append({
                                'content': dt['content'],
                                'source': dt['filename'],
                                'doc_id': dt['doc_id'],
                                'chunk_index': dt['chunk_index'],
                                'page': dt.get('page'),
                                'relevance_score': float(score),
                                'metadata': meta
                            })
                    logger.info(f"[Doc-Scoped][Session] Retrieved {len(results)} for doc_id={doc_id} query='{query[:50]}...'")
                    results.sort(key=lambda x: x['relevance_score'], reverse=True)
                    return results

                # Ensure per-doc index exists (rebuild if necessary from stored chunks)
                if doc_id not in self.doc_indices:
                    chunks = [dt for dt in self.document_texts if dt['doc_id'] == doc_id]
                    if not chunks:
                        logger.info(f"[Doc-Scoped] No chunks for doc_id={doc_id}")
                        return []
                    texts = [dt['content'] for dt in chunks]
                    emb = self.embedding_model.encode(texts, normalize_embeddings=True).astype('float32')
                    doc_index = faiss.IndexFlatIP(emb.shape[1])
                    doc_index.add(emb)
                    self.doc_embeddings[doc_id] = emb
                    self.doc_chunks[doc_id] = chunks
                    self.doc_indices[doc_id] = doc_index

                doc_index = self.doc_indices[doc_id]
                emb = self.doc_embeddings[doc_id]
                scores, indices = doc_index.search(query_embedding, min(top_k, emb.shape[0]))
                chunks = self.doc_chunks.get(doc_id, [])
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(chunks):
                        dt = chunks[idx]
                        meta = self.documents.get(doc_id, {})
                        results.append({
                            'content': dt['content'],
                            'source': dt['filename'],
                            'doc_id': dt['doc_id'],
                            'chunk_index': dt['chunk_index'],
                            'page': dt.get('page'),
                            'relevance_score': float(score),
                            'metadata': meta
                        })
                logger.info(f"[Doc-Scoped] Retrieved {len(results)} for doc_id={doc_id} query='{query[:50]}...'")
            else:
                # Global retrieval across all documents by iterating per-doc indices
                for did, doc_index in self.doc_indices.items():
                    emb = self.doc_embeddings.get(did)
                    if emb is None or emb.shape[0] == 0:
                        continue
                    scores, indices = doc_index.search(query_embedding, min(top_k, emb.shape[0]))
                    chunks = self.doc_chunks.get(did, [])
                    for score, idx in zip(scores[0], indices[0]):
                        if 0 <= idx < len(chunks):
                            dt = chunks[idx]
                            meta = self.documents.get(did, {})
                            if meta.get('deleted', False):
                                continue
                            results.append({
                                'content': dt['content'],
                                'source': dt['filename'],
                                'doc_id': dt['doc_id'],
                                'chunk_index': dt['chunk_index'],
                                'page': dt.get('page'),
                                'relevance_score': float(score),
                                'metadata': meta
                            })
                logger.info(f"[Global] Retrieved {len(results)} for query='{query[:50]}...'")

            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store."""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        Note: FAISS doesn't support deletion, so we mark documents as deleted.
        """
        try:
            if doc_id in self.documents:
                # Mark as deleted in metadata
                self.documents[doc_id]['deleted'] = True
                self.documents[doc_id]['deleted_at'] = str(np.datetime64('now'))
                
                # Remove from document texts
                self.document_texts = [dt for dt in self.document_texts if dt['doc_id'] != doc_id]
                
                # Save changes
                self._save_metadata()
                
                logger.info(f"Marked document {doc_id} as deleted")
                return True
            else:
                logger.warning(f"Document {doc_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        total_docs = len(self.documents)
        active_docs = len([d for d in self.documents.values() if not d.get('deleted', False)])
        total_chunks = len(self.document_texts)
        total_size = sum(d.get('text_length', 0) for d in self.documents.values() if not d.get('deleted', False))
        
        return {
            'total_documents': total_docs,
            'active_documents': active_docs,
            'deleted_documents': total_docs - active_docs,
            'total_chunks': total_chunks,
            'total_size_bytes': total_size,
            'index_size': self.index.ntotal if self.index else 0
        }
