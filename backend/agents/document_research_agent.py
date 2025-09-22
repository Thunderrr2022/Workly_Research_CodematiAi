"""
DocumentResearchAgent - Independent document analysis and research
Analyzes uploaded documents independently and generates comprehensive insights.
"""

import logging
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from .retriever_agent import RetrieverAgent
from .synthesizer_agent import SynthesizerAgent
from .reasoning_logger import ReasoningLogger
from .gap_detector import KnowledgeGapDetector

logger = logging.getLogger(__name__)

class DocumentResearchAgent:
    """
    Independent document research agent that analyzes uploaded documents
    and generates comprehensive insights without requiring user queries.
    """
    
    def __init__(self, retriever_agent=None, synthesizer_agent=None, reasoning_logger=None, gap_detector=None):
        """Initialize the DocumentResearchAgent."""
        self.retriever_agent = retriever_agent or RetrieverAgent()
        self.synthesizer_agent = synthesizer_agent or SynthesizerAgent()
        self.reasoning_logger = reasoning_logger or ReasoningLogger()
        self.gap_detector = gap_detector or KnowledgeGapDetector()
        
        logger.info("DocumentResearchAgent initialized")
    
    def research_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive research on a specific document.
        
        Args:
            doc_id: Document ID to research
            
        Returns:
            Research results with insights, reasoning trace, and gaps
        """
        try:
            # Start new research session
            session_id = self.reasoning_logger.start_session()
            self.reasoning_logger.log_step(
                "Document Research Started",
                f"Beginning comprehensive analysis of document {doc_id}"
            )
            
            # Get document information
            documents = self.retriever_agent.list_documents()
            target_doc = None
            for doc in documents:
                if doc['id'] == doc_id and not doc.get('deleted', False):
                    target_doc = doc
                    break
            
            if not target_doc:
                self.reasoning_logger.log_error(
                    "Document Not Found",
                    f"Document {doc_id} not found or has been deleted"
                )
                return {
                    "error": "Document not found",
                    "reasoning_trace": self.reasoning_logger.get_trace(),
                    "gaps": ["Document not available for analysis"],
                    "insights": []
                }
            
            self.reasoning_logger.log_step(
                "Document Identified",
                f"Found document: {target_doc['filename']} ({target_doc['text_length']} characters, {target_doc['chunk_count']} chunks)"
            )
            
            # Generate research questions for the document
            research_questions = self._generate_document_research_questions(target_doc)
            self.reasoning_logger.log_step(
                "Research Questions Generated",
                f"Created {len(research_questions)} research questions for comprehensive document analysis"
            )
            
            # Retrieve relevant sections for each research question (only from this document)
            all_retrieved_docs = []
            for question in research_questions:
                docs = self.retriever_agent.retrieve_documents(question, top_k=5, doc_id=doc_id)
                all_retrieved_docs.extend(docs)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                if doc['content'] not in seen:
                    seen.add(doc['content'])
                    unique_docs.append(doc)
            
            self.reasoning_logger.log_step(
                "Document Sections Retrieved",
                f"Retrieved {len(unique_docs)} relevant document sections across {len(research_questions)} research questions"
            )
            
            # Generate comprehensive insights
            insights = self._generate_document_insights(target_doc, unique_docs, research_questions)
            self.reasoning_logger.log_step(
                "Insights Generated",
                f"Generated {len(insights)} comprehensive insights about the document"
            )
            
            # Generate comprehensive answer using synthesizer
            if unique_docs:
                answer = self.synthesizer_agent.synthesize(
                    f"Comprehensive analysis of {target_doc['filename']}",
                    unique_docs,
                    research_questions
                )
            else:
                answer = f"# Document Analysis: {target_doc['filename']}\n\n## Summary\nThis document contains {target_doc['text_length']} characters across {target_doc['chunk_count']} sections. However, no specific content could be retrieved for detailed analysis.\n\n## Document Information\n- **Filename**: {target_doc['filename']}\n- **Size**: {target_doc['text_length']} characters\n- **Sections**: {target_doc['chunk_count']} chunks\n- **Uploaded**: {target_doc['added_at']}\n\n## Analysis Status\nDocument has been successfully processed and indexed, but detailed content analysis requires additional processing."
            
            self.reasoning_logger.log_step(
                "Answer Generated",
                f"Generated comprehensive answer for document analysis"
            )
            
            # Detect knowledge gaps
            gaps = self.gap_detector.detect_gaps(
                f"Comprehensive analysis of {target_doc['filename']}",
                unique_docs,
                research_questions
            )
            self.reasoning_logger.log_step(
                "Knowledge Gaps Identified",
                f"Identified {len(gaps)} knowledge gaps in the document analysis"
            )
            
            # Get reasoning trace
            reasoning_trace = self.reasoning_logger.get_trace()
            
            return {
                "answer": answer,
                "document_info": target_doc,
                "research_questions": research_questions,
                "insights": insights,
                "reasoning_trace": reasoning_trace,
                "gaps": gaps,
                "retrieved_sections": len(unique_docs),
                "session_id": session_id
            }
            
        except Exception as e:
            self.reasoning_logger.log_error(
                "Document Research Error",
                f"Error during document research: {str(e)}"
            )
            logger.error(f"Error in document research: {e}")
            return {
                "error": f"Error during document research: {str(e)}",
                "reasoning_trace": self.reasoning_logger.get_trace(),
                "gaps": ["Error occurred during document analysis"],
                "insights": []
            }
    
    def _generate_document_research_questions(self, document: Dict[str, Any]) -> List[str]:
        """Generate comprehensive research questions for document analysis."""
        filename = document['filename'].lower()
        text_length = document['text_length']
        
        # Base research questions
        base_questions = [
            "What is the main topic and purpose of this document?",
            "What are the key concepts and ideas presented?",
            "What are the main findings or conclusions?",
            "What methodology or approach is used?",
            "What are the implications and applications?"
        ]
        
        # Context-specific questions based on filename and content
        context_questions = []
        
        if any(keyword in filename for keyword in ['research', 'study', 'analysis']):
            context_questions.extend([
                "What research methodology was used?",
                "What are the research findings and results?",
                "What are the limitations of this research?",
                "What are the recommendations for future research?"
            ])
        
        if any(keyword in filename for keyword in ['report', 'summary', 'overview']):
            context_questions.extend([
                "What are the main points and key takeaways?",
                "What data and evidence is presented?",
                "What are the conclusions and recommendations?",
                "What are the next steps or action items?"
            ])
        
        if any(keyword in filename for keyword in ['guide', 'manual', 'instructions']):
            context_questions.extend([
                "What processes or procedures are described?",
                "What are the step-by-step instructions?",
                "What tools or resources are required?",
                "What are the best practices and tips?"
            ])
        
        if any(keyword in filename for keyword in ['policy', 'regulation', 'compliance']):
            context_questions.extend([
                "What are the policy requirements and regulations?",
                "What are the compliance obligations?",
                "What are the penalties or consequences?",
                "What are the implementation guidelines?"
            ])
        
        # Technical document questions
        if any(keyword in filename for keyword in ['technical', 'specification', 'api', 'code']):
            context_questions.extend([
                "What are the technical specifications and requirements?",
                "What are the system architecture and components?",
                "What are the implementation details?",
                "What are the technical challenges and solutions?"
            ])
        
        # Combine and return questions
        all_questions = base_questions + context_questions
        
        # Limit to reasonable number based on document size
        if text_length < 5000:
            return all_questions[:6]  # Shorter documents get fewer questions
        elif text_length < 20000:
            return all_questions[:8]
        else:
            return all_questions[:10]  # Longer documents get more comprehensive analysis
    
    def _generate_document_insights(self, document: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], research_questions: List[str]) -> List[Dict[str, Any]]:
        """Generate comprehensive insights about the document."""
        insights = []
        
        # Document overview insight
        insights.append({
            "type": "Document Overview",
            "title": "Document Summary",
            "content": f"This document ({document['filename']}) contains {document['text_length']} characters across {document['chunk_count']} sections. It was uploaded on {document['added_at']} and has been processed for comprehensive analysis.",
            "confidence": "High",
            "sources": [document['filename']]
        })
        
        # Content analysis insights
        if retrieved_docs:
            # Key topics insight
            topics = self._extract_key_topics(retrieved_docs)
            if topics:
                insights.append({
                    "type": "Content Analysis",
                    "title": "Key Topics and Themes",
                    "content": f"The document covers several key topics: {', '.join(topics[:5])}. These themes appear throughout the document and represent the main areas of focus.",
                    "confidence": "High",
                    "sources": [doc['source'] for doc in retrieved_docs[:3]]
                })
            
            # Technical complexity insight
            complexity = self._assess_technical_complexity(retrieved_docs)
            insights.append({
                "type": "Content Analysis",
                "title": "Technical Complexity",
                "content": f"The document has {complexity['level']} technical complexity. {complexity['description']}",
                "confidence": "Medium",
                "sources": [doc['source'] for doc in retrieved_docs[:2]]
            })
            
            # Information density insight
            density = self._assess_information_density(retrieved_docs)
            insights.append({
                "type": "Content Analysis",
                "title": "Information Density",
                "content": f"The document has {density['level']} information density. {density['description']}",
                "confidence": "Medium",
                "sources": [doc['source'] for doc in retrieved_docs[:2]]
            })
        
        # Research question insights
        for i, question in enumerate(research_questions[:5], 1):
            relevant_docs = [doc for doc in retrieved_docs if self._is_relevant_to_question(doc, question)]
            if relevant_docs:
                insight_content = self._generate_question_insight(question, relevant_docs)
                insights.append({
                    "type": "Research Finding",
                    "title": f"Research Question {i}",
                    "content": insight_content,
                    "confidence": "High" if len(relevant_docs) > 2 else "Medium",
                    "sources": [doc['source'] for doc in relevant_docs[:3]]
                })
        
        return insights
    
    def _extract_key_topics(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract key topics from document content."""
        # Simple keyword extraction based on frequency
        word_freq = {}
        for doc in documents:
            content = doc.get('content', '').lower()
            # Remove common words and extract meaningful terms
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content)
            for word in words:
                if word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'there', 'could', 'other', 'after', 'first', 'well', 'also', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can']:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq > 1]
    
    def _assess_technical_complexity(self, documents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Assess the technical complexity of the document."""
        technical_indicators = ['algorithm', 'implementation', 'architecture', 'system', 'protocol', 'framework', 'api', 'database', 'software', 'hardware', 'network', 'security', 'performance', 'optimization', 'configuration', 'deployment']
        
        total_content = ' '.join([doc.get('content', '') for doc in documents]).lower()
        technical_count = sum(1 for indicator in technical_indicators if indicator in total_content)
        
        if technical_count > 10:
            return {"level": "High", "description": "The document contains significant technical content with detailed technical specifications, methodologies, and implementation details."}
        elif technical_count > 5:
            return {"level": "Medium", "description": "The document includes moderate technical content with some technical concepts and procedures."}
        else:
            return {"level": "Low", "description": "The document is primarily non-technical with minimal technical terminology or concepts."}
    
    def _assess_information_density(self, documents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Assess the information density of the document."""
        total_chars = sum(len(doc.get('content', '')) for doc in documents)
        total_docs = len(documents)
        
        if total_docs == 0:
            return {"level": "Unknown", "description": "No content available for analysis."}
        
        avg_chars_per_section = total_chars / total_docs
        
        if avg_chars_per_section > 1000:
            return {"level": "High", "description": "The document contains dense information with detailed explanations and comprehensive coverage of topics."}
        elif avg_chars_per_section > 500:
            return {"level": "Medium", "description": "The document has moderate information density with balanced detail and coverage."}
        else:
            return {"level": "Low", "description": "The document contains concise information with brief explanations and summaries."}
    
    def _is_relevant_to_question(self, doc: Dict[str, Any], question: str) -> bool:
        """Check if document is relevant to a specific question."""
        content = doc.get('content', '').lower()
        question_words = question.lower().split()
        
        # Check for keyword matches
        significant_words = [word for word in question_words if len(word) > 3]
        matches = sum(1 for word in significant_words if word in content)
        
        return matches >= len(significant_words) * 0.3
    
    def _generate_question_insight(self, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate insight for a specific research question."""
        if not relevant_docs:
            return f"No specific information found to answer: {question}"
        
        # Combine relevant content
        combined_content = ' '.join([doc.get('content', '') for doc in relevant_docs[:3]])
        
        # Extract key points (first sentence of each relevant section)
        key_points = []
        for doc in relevant_docs[:3]:
            content = doc.get('content', '')
            first_sentence = content.split('.')[0] if '.' in content else content[:200]
            if first_sentence.strip():
                key_points.append(first_sentence.strip())
        
        if key_points:
            return f"Based on the document analysis: {' '.join(key_points[:2])}... This addresses the question: {question}"
        else:
            return f"The document contains relevant information about: {question}. The content covers this topic across multiple sections."
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get statistics about document research capabilities."""
        doc_stats = self.retriever_agent.get_document_stats()
        return {
            "total_documents": doc_stats['active_documents'],
            "total_chunks": doc_stats['total_chunks'],
            "vector_store_size": doc_stats['index_size'],
            "research_capabilities": [
                "Document Overview Analysis",
                "Key Topic Extraction",
                "Technical Complexity Assessment",
                "Information Density Analysis",
                "Research Question Generation",
                "Comprehensive Insight Generation"
            ]
        }
