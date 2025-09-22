"""
KnowledgeGapDetector - Missing information identification system
Detects knowledge gaps and missing information in research results.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

class KnowledgeGapDetector:
    """
    Intelligent gap detection agent that identifies missing information
    and knowledge gaps in research results.
    """
    
    def __init__(self):
        """Initialize the KnowledgeGapDetector with LLM configuration."""
        self.client = None
        self.model = config.OPENAI_MODEL
        
        # Initialize OpenAI client if API key is available
        if config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            logger.info(f"KnowledgeGapDetector initialized with OpenAI model: {self.model}")
        else:
            logger.warning("No OpenAI API key provided. Using rule-based gap detection.")
        
        # Common gap indicators
        self.gap_indicators = [
            "no data found",
            "insufficient information",
            "limited data",
            "not available",
            "unclear",
            "unknown",
            "requires further research",
            "needs more investigation",
            "lacks information",
            "missing data",
            "incomplete information",
            "not specified",
            "not mentioned",
            "no evidence",
            "no research",
            "no studies",
            "no documentation"
        ]
        
        # Common knowledge domains that might have gaps
        self.knowledge_domains = [
            "statistical data",
            "quantitative analysis",
            "longitudinal studies",
            "comparative analysis",
            "regulatory information",
            "implementation details",
            "cost analysis",
            "timeline information",
            "geographic coverage",
            "demographic breakdown",
            "technical specifications",
            "performance metrics",
            "safety data",
            "efficacy studies",
            "market analysis"
        ]
    
    def _detect_gaps_with_llm(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> List[str]:
        """Detect knowledge gaps using LLM analysis."""
        try:
            # Prepare document summary
            doc_summary = self._create_document_summary(documents)
            
            prompt = f"""You are an expert research analyst tasked with identifying knowledge gaps in research results.

RESEARCH QUESTION: {query}

SUB-QUESTIONS ADDRESSED:
{chr(10).join(f"- {sq}" for sq in sub_queries)}

DOCUMENTS ANALYZED:
{doc_summary}

INSTRUCTIONS:
1. Analyze the provided documents and sub-questions
2. Identify specific areas where information is missing, incomplete, or insufficient
3. Focus on gaps that would be important for a comprehensive answer
4. Consider different aspects: quantitative data, qualitative insights, temporal coverage, geographic scope, etc.
5. Be specific about what information is missing
6. Format each gap as a clear, actionable statement

GAP DETECTION CRITERIA:
- Missing statistical or quantitative data
- Lack of recent information (if relevant)
- Insufficient geographic or demographic coverage
- Missing comparative analysis
- Lack of implementation details
- Missing cost or resource information
- Insufficient evidence for claims
- Limited scope of analysis

FORMAT: Return only the identified gaps, one per line, without numbering or bullets.

EXAMPLE OUTPUT:
No longitudinal data on patient outcomes beyond 2-year follow-up periods
Insufficient information on regulatory compliance variations across different international healthcare systems
Limited quantitative analysis of cost-effectiveness compared to traditional methods
Missing data on implementation challenges in rural healthcare settings

Now, please identify the knowledge gaps in this research:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst specializing in identifying knowledge gaps and missing information in research results. Be thorough but concise in your gap analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            gaps_text = response.choices[0].message.content.strip()
            
            # Parse gaps from response
            gaps = []
            for line in gaps_text.split('\n'):
                line = line.strip()
                if line and not line.startswith(('Research', 'Sub-questions', 'Documents', 'Instructions', 'Gap', 'Format', 'Example')):
                    # Remove numbering if present
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^[-*]\s*', '', line)
                    if line and len(line) > 20:  # Ensure substantial gaps
                        gaps.append(line)
            
            logger.info(f"LLM detected {len(gaps)} knowledge gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error in LLM gap detection: {e}")
            return self._detect_gaps_with_rules(query, documents, sub_queries)
    
    def _detect_gaps_with_rules(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> List[str]:
        """Fallback rule-based gap detection."""
        gaps = []
        
        # Check for low document coverage
        if len(documents) < 3:
            gaps.append("Limited document coverage - only a few sources available for analysis")
        
        # Check for low relevance scores
        low_relevance_docs = [doc for doc in documents if doc.get('relevance_score', 0) < 0.5]
        if len(low_relevance_docs) > len(documents) * 0.5:
            gaps.append("Many retrieved documents have low relevance scores, indicating potential information gaps")
        
        # Check for missing quantitative data
        has_quantitative = any(self._contains_quantitative_data(doc.get('content', '')) for doc in documents)
        if not has_quantitative:
            gaps.append("No quantitative data or statistical analysis found in the retrieved documents")
        
        # Check for temporal coverage
        has_recent_info = any(self._contains_recent_information(doc.get('content', '')) for doc in documents)
        if not has_recent_info:
            gaps.append("Limited recent information - documents may not reflect current developments")
        
        # Check for geographic coverage
        has_geographic_info = any(self._contains_geographic_data(doc.get('content', '')) for doc in documents)
        if not has_geographic_info:
            gaps.append("No geographic or regional analysis found in the available documents")
        
        # Check for comparative analysis
        has_comparative = any(self._contains_comparative_analysis(doc.get('content', '')) for doc in documents)
        if not has_comparative:
            gaps.append("No comparative analysis or benchmarking data found")
        
        # Check for implementation details
        has_implementation = any(self._contains_implementation_details(doc.get('content', '')) for doc in documents)
        if not has_implementation:
            gaps.append("Limited implementation details or practical guidance found")
        
        # Check for cost/economic analysis
        has_economic = any(self._contains_economic_data(doc.get('content', '')) for doc in documents)
        if not has_economic:
            gaps.append("No cost analysis or economic impact data found")
        
        # Check for regulatory information
        has_regulatory = any(self._contains_regulatory_info(doc.get('content', '')) for doc in documents)
        if not has_regulatory:
            gaps.append("No regulatory or compliance information found")
        
        logger.info(f"Rule-based detection found {len(gaps)} knowledge gaps")
        return gaps
    
    def _create_document_summary(self, documents: List[Dict[str, Any]]) -> str:
        """Create a summary of retrieved documents for gap analysis."""
        if not documents:
            return "No documents were retrieved for analysis."
        
        summary_parts = []
        sources = set(doc.get('source', 'Unknown') for doc in documents)
        
        summary_parts.append(f"Total documents: {len(documents)}")
        summary_parts.append(f"Sources: {', '.join(sources)}")
        
        # Add content snippets
        for i, doc in enumerate(documents[:5], 1):  # Limit to first 5 docs
            content = doc.get('content', '')[:200]  # First 200 chars
            source = doc.get('source', 'Unknown')
            relevance = doc.get('relevance_score', 0)
            summary_parts.append(f"Doc {i} ({source}, relevance: {relevance:.2f}): {content}...")
        
        if len(documents) > 5:
            summary_parts.append(f"... and {len(documents) - 5} more documents")
        
        return "\n".join(summary_parts)
    
    def _contains_quantitative_data(self, content: str) -> bool:
        """Check if content contains quantitative data."""
        # Look for numbers, percentages, statistics
        patterns = [
            r'\d+%',  # Percentages
            r'\d+\.\d+',  # Decimals
            r'\$\d+',  # Money
            r'\d+\s*(million|billion|thousand)',  # Large numbers
            r'statistical|statistics|data|analysis|study|research'
        ]
        
        return any(re.search(pattern, content.lower()) for pattern in patterns)
    
    def _contains_recent_information(self, content: str) -> bool:
        """Check if content contains recent information."""
        # Look for recent years
        current_year = 2024
        years = re.findall(r'\b(20\d{2})\b', content)
        if years:
            recent_years = [year for year in years if int(year) >= current_year - 3]
            return len(recent_years) > 0
        
        # Look for recent indicators
        recent_indicators = ['recent', 'latest', 'current', 'new', 'updated', '2024', '2023', '2022']
        return any(indicator in content.lower() for indicator in recent_indicators)
    
    def _contains_geographic_data(self, content: str) -> bool:
        """Check if content contains geographic information."""
        geographic_indicators = [
            'country', 'countries', 'region', 'regions', 'global', 'international',
            'united states', 'europe', 'asia', 'africa', 'america', 'china', 'india',
            'geographic', 'geographical', 'worldwide', 'nationwide'
        ]
        
        return any(indicator in content.lower() for indicator in geographic_indicators)
    
    def _contains_comparative_analysis(self, content: str) -> bool:
        """Check if content contains comparative analysis."""
        comparative_indicators = [
            'compare', 'comparison', 'versus', 'vs', 'compared to', 'relative to',
            'benchmark', 'benchmarking', 'alternative', 'different', 'similar',
            'contrast', 'difference', 'similarity'
        ]
        
        return any(indicator in content.lower() for indicator in comparative_indicators)
    
    def _contains_implementation_details(self, content: str) -> bool:
        """Check if content contains implementation details."""
        implementation_indicators = [
            'implementation', 'deploy', 'deployment', 'rollout', 'adoption',
            'process', 'procedure', 'steps', 'guidelines', 'best practices',
            'how to', 'methodology', 'approach', 'strategy'
        ]
        
        return any(indicator in content.lower() for indicator in implementation_indicators)
    
    def _contains_economic_data(self, content: str) -> bool:
        """Check if content contains economic data."""
        economic_indicators = [
            'cost', 'price', 'budget', 'economic', 'financial', 'investment',
            'roi', 'return on investment', 'expense', 'revenue', 'profit',
            'affordable', 'expensive', 'cheap', 'value', 'worth'
        ]
        
        return any(indicator in content.lower() for indicator in economic_indicators)
    
    def _contains_regulatory_info(self, content: str) -> bool:
        """Check if content contains regulatory information."""
        regulatory_indicators = [
            'regulation', 'regulatory', 'compliance', 'legal', 'law', 'policy',
            'fda', 'approval', 'certification', 'standard', 'guideline',
            'government', 'federal', 'state', 'local', 'authority'
        ]
        
        return any(indicator in content.lower() for indicator in regulatory_indicators)
    
    def _categorize_gaps(self, gaps: List[str]) -> Dict[str, List[str]]:
        """Categorize gaps by type."""
        categorized = {
            "Data Quality": [],
            "Coverage": [],
            "Temporal": [],
            "Geographic": [],
            "Methodological": [],
            "Regulatory": [],
            "Economic": [],
            "Other": []
        }
        
        for gap in gaps:
            gap_lower = gap.lower()
            
            if any(word in gap_lower for word in ['data', 'information', 'evidence', 'research']):
                categorized["Data Quality"].append(gap)
            elif any(word in gap_lower for word in ['coverage', 'scope', 'limited', 'insufficient']):
                categorized["Coverage"].append(gap)
            elif any(word in gap_lower for word in ['recent', 'current', 'timeline', 'longitudinal']):
                categorized["Temporal"].append(gap)
            elif any(word in gap_lower for word in ['geographic', 'regional', 'country', 'international']):
                categorized["Geographic"].append(gap)
            elif any(word in gap_lower for word in ['method', 'analysis', 'study', 'research']):
                categorized["Methodological"].append(gap)
            elif any(word in gap_lower for word in ['regulatory', 'compliance', 'legal', 'policy']):
                categorized["Regulatory"].append(gap)
            elif any(word in gap_lower for word in ['cost', 'economic', 'financial', 'budget']):
                categorized["Economic"].append(gap)
            else:
                categorized["Other"].append(gap)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def detect_gaps(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> List[str]:
        """
        Detect knowledge gaps in the research results.
        
        Args:
            query: Original research query
            documents: List of retrieved document chunks
            sub_queries: List of sub-questions derived from the main query
            
        Returns:
            List of identified knowledge gaps
        """
        try:
            logger.info(f"Detecting gaps for query: {query[:50]}... with {len(documents)} documents")
            
            # Choose detection method
            if self.client:
                gaps = self._detect_gaps_with_llm(query, documents, sub_queries)
            else:
                gaps = self._detect_gaps_with_rules(query, documents, sub_queries)
            
            # Remove duplicates while preserving order
            unique_gaps = []
            seen = set()
            for gap in gaps:
                if gap not in seen:
                    seen.add(gap)
                    unique_gaps.append(gap)
            
            # Limit number of gaps
            unique_gaps = unique_gaps[:10]  # Maximum 10 gaps
            
            logger.info(f"Detected {len(unique_gaps)} knowledge gaps")
            return unique_gaps
            
        except Exception as e:
            logger.error(f"Error detecting gaps: {e}")
            return ["Error occurred during gap detection - unable to identify specific knowledge gaps"]
    
    def get_gap_analysis_stats(self, query: str, documents: List[Dict[str, Any]], gaps: List[str]) -> Dict[str, Any]:
        """Get statistics about the gap analysis process."""
        categorized_gaps = self._categorize_gaps(gaps)
        
        return {
            "query": query,
            "document_count": len(documents),
            "total_gaps": len(gaps),
            "categorized_gaps": categorized_gaps,
            "gap_categories": list(categorized_gaps.keys()),
            "detection_method": "LLM" if self.client else "Rule-based",
            "average_document_relevance": sum(doc.get('relevance_score', 0) for doc in documents) / len(documents) if documents else 0
        }
