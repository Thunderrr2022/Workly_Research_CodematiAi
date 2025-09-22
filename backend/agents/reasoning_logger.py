"""
ReasoningLogger - Chain-of-thought transparency system
Logs and tracks the reasoning process for full transparency in research.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ReasoningLogger:
    """
    Comprehensive reasoning logger that tracks the entire research process
    to provide full chain-of-thought transparency.
    """
    
    def __init__(self):
        """Initialize the ReasoningLogger."""
        self.reasoning_steps = []
        self.current_session_id = None
        self.start_time = None
        
        logger.info("ReasoningLogger initialized")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new reasoning session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id:
            self.current_session_id = session_id
        else:
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.start_time = datetime.now()
        self.reasoning_steps = []
        
        self.log_step("Session Started", f"Research session {self.current_session_id} initiated")
        
        logger.info(f"Started reasoning session: {self.current_session_id}")
        return self.current_session_id
    
    def log_step(self, step_type: str, description: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a reasoning step.
        
        Args:
            step_type: Type of step (e.g., "Query Decomposition", "Document Retrieval")
            description: Detailed description of the step
            metadata: Optional additional metadata
        """
        step = {
            "step_number": len(self.reasoning_steps) + 1,
            "step_type": step_type,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.reasoning_steps.append(step)
        
        logger.info(f"Reasoning step {step['step_number']}: {step_type} - {description}")
    
    def log_query_decomposition(self, original_query: str, sub_queries: List[str], method: str):
        """Log query decomposition step."""
        self.log_step(
            "Query Decomposition",
            f"Broke down the query '{original_query}' into {len(sub_queries)} focused sub-questions using {method} method",
            {
                "original_query": original_query,
                "sub_queries": sub_queries,
                "decomposition_method": method,
                "sub_query_count": len(sub_queries)
            }
        )
    
    def log_document_retrieval(self, sub_query: str, retrieved_docs: List[Dict[str, Any]], total_docs: int):
        """Log document retrieval step."""
        sources = list(set(doc.get('source', 'Unknown') for doc in retrieved_docs))
        avg_relevance = sum(doc.get('relevance_score', 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
        
        self.log_step(
            "Document Retrieval",
            f"Retrieved {len(retrieved_docs)} relevant document sections for sub-query: '{sub_query}' from {len(sources)} sources",
            {
                "sub_query": sub_query,
                "retrieved_count": len(retrieved_docs),
                "total_available_docs": total_docs,
                "sources": sources,
                "average_relevance_score": round(avg_relevance, 3),
                "top_relevance_score": max((doc.get('relevance_score', 0) for doc in retrieved_docs), default=0)
            }
        )
    
    def log_synthesis(self, query: str, document_count: int, synthesis_method: str, answer_length: int):
        """Log information synthesis step."""
        self.log_step(
            "Information Synthesis",
            f"Synthesized comprehensive answer from {document_count} document sections using {synthesis_method}",
            {
                "query": query,
                "document_count": document_count,
                "synthesis_method": synthesis_method,
                "answer_length": answer_length,
                "synthesis_timestamp": datetime.now().isoformat()
            }
        )
    
    def log_gap_analysis(self, gaps: List[str], gap_analysis_method: str):
        """Log knowledge gap analysis step."""
        self.log_step(
            "Knowledge Gap Analysis",
            f"Identified {len(gaps)} knowledge gaps in the research using {gap_analysis_method}",
            {
                "gap_count": len(gaps),
                "gaps": gaps,
                "analysis_method": gap_analysis_method
            }
        )
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Log an error that occurred during processing."""
        self.log_step(
            "Error",
            f"{error_type}: {error_message}",
            {
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
                "is_error": True
            }
        )
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics."""
        self.log_step(
            "Performance Metric",
            f"{metric_name}: {value} {unit}".strip(),
            {
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "is_metric": True
            }
        )
    
    def get_trace(self) -> List[str]:
        """
        Get the reasoning trace as a list of formatted step descriptions.
        
        Returns:
            List of formatted reasoning step descriptions
        """
        trace = []
        
        for step in self.reasoning_steps:
            # Skip error steps and metrics from the main trace
            if step.get("metadata", {}).get("is_error", False) or step.get("metadata", {}).get("is_metric", False):
                continue
            
            # Format the step description
            step_type = step["step_type"]
            description = step["description"]
            
            # Create a more readable format
            if step_type == "Query Decomposition":
                trace.append(f"Step {step['step_number']}: {description}")
            elif step_type == "Document Retrieval":
                trace.append(f"Step {step['step_number']}: {description}")
            elif step_type == "Information Synthesis":
                trace.append(f"Step {step['step_number']}: {description}")
            elif step_type == "Knowledge Gap Analysis":
                trace.append(f"Step {step['step_number']}: {description}")
            else:
                trace.append(f"Step {step['step_number']}: {step_type} - {description}")
        
        return trace
    
    def get_detailed_trace(self) -> List[Dict[str, Any]]:
        """
        Get the complete detailed reasoning trace.
        
        Returns:
            List of detailed reasoning steps with all metadata
        """
        return self.reasoning_steps.copy()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current reasoning session.
        
        Returns:
            Session summary with statistics
        """
        if not self.reasoning_steps:
            return {"message": "No reasoning steps recorded"}
        
        # Calculate session duration
        duration = None
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        
        # Count different step types
        step_types = {}
        for step in self.reasoning_steps:
            step_type = step["step_type"]
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        # Count errors
        error_count = sum(1 for step in self.reasoning_steps 
                         if step.get("metadata", {}).get("is_error", False))
        
        # Count metrics
        metric_count = sum(1 for step in self.reasoning_steps 
                          if step.get("metadata", {}).get("is_metric", False))
        
        return {
            "session_id": self.current_session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": duration,
            "total_steps": len(self.reasoning_steps),
            "step_types": step_types,
            "error_count": error_count,
            "metric_count": metric_count,
            "last_step_time": self.reasoning_steps[-1]["timestamp"] if self.reasoning_steps else None
        }
    
    def export_trace(self, format: str = "json") -> str:
        """
        Export the reasoning trace in various formats.
        
        Args:
            format: Export format ("json", "text", "markdown")
            
        Returns:
            Exported trace as string
        """
        if format.lower() == "json":
            return json.dumps({
                "session_summary": self.get_session_summary(),
                "reasoning_steps": self.get_detailed_trace()
            }, indent=2)
        
        elif format.lower() == "text":
            lines = []
            lines.append("=== REASONING TRACE ===")
            lines.append(f"Session ID: {self.current_session_id}")
            lines.append(f"Start Time: {self.start_time}")
            lines.append("")
            
            for step in self.reasoning_steps:
                lines.append(f"Step {step['step_number']}: {step['step_type']}")
                lines.append(f"  Description: {step['description']}")
                lines.append(f"  Timestamp: {step['timestamp']}")
                if step.get("metadata"):
                    lines.append(f"  Metadata: {json.dumps(step['metadata'], indent=4)}")
                lines.append("")
            
            return "\n".join(lines)
        
        elif format.lower() == "markdown":
            lines = []
            lines.append("# Reasoning Trace")
            lines.append(f"**Session ID:** {self.current_session_id}")
            lines.append(f"**Start Time:** {self.start_time}")
            lines.append("")
            
            for step in self.reasoning_steps:
                lines.append(f"## Step {step['step_number']}: {step['step_type']}")
                lines.append(f"**Description:** {step['description']}")
                lines.append(f"**Timestamp:** {step['timestamp']}")
                if step.get("metadata"):
                    lines.append("**Metadata:**")
                    lines.append("```json")
                    lines.append(json.dumps(step['metadata'], indent=2))
                    lines.append("```")
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_session(self):
        """Clear the current reasoning session."""
        self.reasoning_steps = []
        self.current_session_id = None
        self.start_time = None
        logger.info("Reasoning session cleared")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from the reasoning trace."""
        metrics = {}
        
        for step in self.reasoning_steps:
            if step.get("metadata", {}).get("is_metric", False):
                metric_name = step["metadata"].get("metric_name")
                value = step["metadata"].get("value")
                unit = step["metadata"].get("unit", "")
                
                if metric_name:
                    metrics[metric_name] = {
                        "value": value,
                        "unit": unit,
                        "timestamp": step["timestamp"]
                    }
        
        return metrics


