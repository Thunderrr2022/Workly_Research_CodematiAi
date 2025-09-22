"""
QuerySplitter - Dynamic query decomposition system
Breaks down complex queries into focused sub-questions for better research coverage.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

class QuerySplitter:
    """
    Intelligent query decomposition agent that breaks down complex research questions
    into focused sub-questions for comprehensive analysis.
    """
    
    def __init__(self):
        """Initialize the QuerySplitter with LLM configuration."""
        self.client = None
        self.model = config.OPENAI_MODEL
        
        # Initialize OpenAI client if API key is available
        if config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            logger.info(f"QuerySplitter initialized with OpenAI model: {self.model}")
        else:
            logger.warning("No OpenAI API key provided. Using rule-based query splitting.")
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex enough to warrant decomposition."""
        # Check for multiple question indicators
        question_indicators = ['?', 'how', 'what', 'why', 'when', 'where', 'which', 'who']
        question_count = sum(1 for indicator in question_indicators if indicator.lower() in query.lower())
        
        # Check for multiple topics (conjunctions)
        conjunctions = ['and', 'or', 'but', 'also', 'additionally', 'furthermore', 'moreover']
        conjunction_count = sum(1 for conj in conjunctions if conj.lower() in query.lower())
        
        # Check query length
        word_count = len(query.split())
        
        # Check for very short queries (like "ar vr") - these should be expanded
        if word_count <= 3 and not any(indicator in query.lower() for indicator in question_indicators):
            return True
        
        # Consider complex if: multiple questions, conjunctions, long query, or very short queries
        return question_count > 1 or conjunction_count > 0 or word_count > 15 or word_count <= 3
    
    def _split_with_llm(self, query: str) -> List[str]:
        """Split query using LLM for intelligent decomposition."""
        try:
            prompt = f"""You are an expert research analyst. Your task is to break down a complex research question into 3-5 focused sub-questions that will help provide comprehensive coverage of the topic.

RESEARCH QUESTION: {query}

INSTRUCTIONS:
1. Identify the main themes and aspects of the question
2. Create 3-5 specific, focused sub-questions
3. Each sub-question should be answerable independently
4. Sub-questions should cover different dimensions (what, how, why, when, where, impact, etc.)
5. Avoid overly broad or vague sub-questions
6. Ensure sub-questions are specific enough to guide targeted research

FORMAT: Return only the sub-questions, one per line, without numbering or bullets.

EXAMPLE:
Original: "How is artificial intelligence transforming healthcare?"
Sub-questions:
What are the main applications of AI in medical diagnosis?
How is AI improving drug discovery processes?
What impact does AI have on patient care and treatment outcomes?
What are the challenges and limitations of AI in healthcare?
How is AI affecting healthcare costs and accessibility?

Now, please break down this research question:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst specializing in query decomposition. Break down complex questions into focused, researchable sub-questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            sub_queries_text = response.choices[0].message.content.strip()
            
            # Parse sub-questions from response
            sub_queries = []
            for line in sub_queries_text.split('\n'):
                line = line.strip()
                if line and not line.startswith(('Original:', 'Sub-questions:', 'Example:')):
                    # Remove numbering if present
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^[-*]\s*', '', line)
                    if line:
                        sub_queries.append(line)
            
            # Limit to maximum number of sub-queries
            sub_queries = sub_queries[:config.MAX_SUB_QUERIES]
            
            logger.info(f"LLM generated {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error in LLM query splitting: {e}")
            return self._split_with_rules(query)
    
    def _split_with_rules(self, query: str) -> List[str]:
        """Fallback rule-based query splitting."""
        query_lower = query.lower()
        sub_queries = []
        
        # Common query patterns and their decompositions
        patterns = {
            # Technology impact patterns
            r'how is (.+) transforming (.+)': [
                f"What are the main applications of {r'\1'} in {r'\2'}?",
                f"How is {r'\1'} improving processes in {r'\2'}?",
                f"What impact does {r'\1'} have on {r'\2'} outcomes?",
                f"What are the challenges of implementing {r'\1'} in {r'\2'}?"
            ],
            
            # Comparison patterns
            r'compare (.+) and (.+)': [
                f"What are the key features of {r'\1'}?",
                f"What are the key features of {r'\2'}?",
                f"What are the main differences between {r'\1'} and {r'\2'}?",
                f"What are the advantages and disadvantages of each approach?"
            ],
            
            # Analysis patterns
            r'analyze (.+)': [
                f"What are the main components of {r'\1'}?",
                f"How does {r'\1'} work?",
                f"What are the benefits and drawbacks of {r'\1'}?",
                f"What are the current trends and future prospects for {r'\1'}?"
            ],
            
            # Impact patterns
            r'what is the impact of (.+) on (.+)': [
                f"How does {r'\1'} affect {r'\2'} directly?",
                f"What are the indirect effects of {r'\1'} on {r'\2'}?",
                f"What are the positive impacts of {r'\1'} on {r'\2'}?",
                f"What are the negative impacts of {r'\1'} on {r'\2'}?"
            ]
        }
        
        # Try to match patterns
        for pattern, templates in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                for template in templates:
                    try:
                        sub_query = template.format(*match.groups())
                        sub_queries.append(sub_query)
                    except:
                        continue
                break
        
        # If no pattern matched, use generic decomposition
        if not sub_queries:
            sub_queries = self._generic_decomposition(query)
        
        # Limit to maximum number of sub-queries
        sub_queries = sub_queries[:config.MAX_SUB_QUERIES]
        
        logger.info(f"Rule-based splitting generated {len(sub_queries)} sub-queries")
        return sub_queries
    
    def _generic_decomposition(self, query: str) -> List[str]:
        """Generic decomposition for queries that don't match specific patterns."""
        # Extract key terms
        words = query.lower().split()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where', 'which', 'who'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 1]
        
        if not key_terms:
            return [query]  # Return original if no key terms found
        
        # Handle special cases for common abbreviations
        expanded_terms = []
        for term in key_terms:
            if term == 'ar':
                expanded_terms.extend(['augmented reality', 'AR'])
            elif term == 'vr':
                expanded_terms.extend(['virtual reality', 'VR'])
            elif term == 'ai':
                expanded_terms.extend(['artificial intelligence', 'AI'])
            elif term == 'ml':
                expanded_terms.extend(['machine learning', 'ML'])
            else:
                expanded_terms.append(term)
        
        # Create generic sub-questions
        sub_queries = []
        
        # What question
        if 'what' not in query.lower():
            sub_queries.append(f"What is {expanded_terms[0]} and how does it work?")
        
        # How question
        if 'how' not in query.lower():
            sub_queries.append(f"How does {expanded_terms[0]} function or operate?")
        
        # Applications question
        sub_queries.append(f"What are the main applications of {expanded_terms[0]}?")
        
        # Current state question
        sub_queries.append(f"What is the current state and development of {expanded_terms[0]}?")
        
        # Future prospects question
        sub_queries.append(f"What are the future prospects and trends for {expanded_terms[0]}?")
        
        return sub_queries
    
    def _validate_sub_queries(self, sub_queries: List[str]) -> List[str]:
        """Validate and clean sub-queries."""
        validated = []
        
        for sub_query in sub_queries:
            # Clean up the sub-query
            sub_query = sub_query.strip()
            
            # Remove extra whitespace
            sub_query = re.sub(r'\s+', ' ', sub_query)
            
            # Ensure it ends with a question mark
            if not sub_query.endswith('?'):
                sub_query += '?'
            
            # Check minimum length
            if len(sub_query) > 10:
                validated.append(sub_query)
        
        return validated
    
    def split_query(self, query: str) -> List[str]:
        """
        Split a complex query into focused sub-questions.
        
        Args:
            query: The original research query
            
        Returns:
            List of focused sub-questions
        """
        try:
            logger.info(f"Splitting query: {query[:50]}...")
            
            # Check if query is complex enough to split
            if not self._is_complex_query(query):
                logger.info("Query is simple, returning as single sub-query")
                return [query]
            
            # Choose splitting method
            if self.client:
                sub_queries = self._split_with_llm(query)
            else:
                sub_queries = self._split_with_rules(query)
            
            # Validate and clean sub-queries
            sub_queries = self._validate_sub_queries(sub_queries)
            
            # Ensure we have at least one sub-query
            if not sub_queries:
                sub_queries = [query]
            
            logger.info(f"Successfully split query into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error splitting query: {e}")
            return [query]  # Return original query as fallback
    
    def get_splitting_stats(self, query: str, sub_queries: List[str]) -> Dict[str, Any]:
        """Get statistics about the query splitting process."""
        return {
            "original_query": query,
            "original_length": len(query.split()),
            "sub_query_count": len(sub_queries),
            "average_sub_query_length": sum(len(sq.split()) for sq in sub_queries) / len(sub_queries) if sub_queries else 0,
            "is_complex": self._is_complex_query(query),
            "splitting_method": "LLM" if self.client else "Rule-based"
        }
