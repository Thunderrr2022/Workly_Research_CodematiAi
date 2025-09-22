"""
SynthesizerAgent - Information synthesis and answer generation
Combines retrieved information into coherent, comprehensive answers using LLM.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import math
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util

from config import config
from transformers import pipeline as hf_pipeline
from openai import OpenAI

logger = logging.getLogger(__name__)

class SynthesizerAgent:
    """
    Advanced synthesis agent that combines information from multiple sources
    to generate comprehensive, well-structured answers.
    """
    
    def __init__(self):
        """Initialize SynthesizerAgent to use deterministic local synthesis only (no external APIs)."""
        self.client = None
        self.model = "local-fallback"
        self.embedding_model: Optional[SentenceTransformer] = None
        self.backend = config.SYNTHESIZER_BACKEND
        if self.backend == "openai" and config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.model = config.OPENAI_MODEL or "gpt-4o-mini"
            logger.info(f"SynthesizerAgent using OpenAI backend: {self.model}")
        elif self.backend == "huggingface" and config.HUGGINGFACE_MODEL:
            self.hf_gen = hf_pipeline("text-generation", model=config.HUGGINGFACE_MODEL)
            logger.info(f"SynthesizerAgent using HuggingFace backend: {config.HUGGINGFACE_MODEL}")
        else:
            self.backend = "local"
            logger.info("SynthesizerAgent using local synthesis backend.")

    # ----------------------------
    # Embeddings + utilities
    # ----------------------------
    def _get_embedder(self) -> SentenceTransformer:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self.embedding_model

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        model = self._get_embedder()
        return model.encode(texts, normalize_embeddings=True).astype('float32')

    # ----------------------------
    # Theming & extraction
    # ----------------------------
    def _cluster_chunks(self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.6) -> List[List[int]]:
        """Greedy clustering of chunks by cosine similarity on embeddings."""
        if not documents:
            return []
        texts = [d.get('content', '') for d in documents]
        emb = self._embed_texts(texts)
        n = emb.shape[0]
        used = [False] * n
        clusters: List[List[int]] = []
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            cluster = [i]
            sims = (emb @ emb[i:i+1].T).reshape(-1)
            for j in range(i + 1, n):
                if not used[j] and sims[j] >= similarity_threshold:
                    used[j] = True
                    cluster.append(j)
            clusters.append(cluster)
        return clusters

    def _label_cluster(self, docs: List[Dict[str, Any]], idxs: List[int]) -> str:
        bag: Dict[str,int] = {}
        for i in idxs:
            text = docs[i].get('content','').lower()
            for w in re.findall(r"[a-zA-Z]{5,}", text):
                bag[w] = bag.get(w,0)+1
        if not bag:
            return "General"
        top = sorted(bag.items(), key=lambda x:x[1], reverse=True)[:5]
        return ", ".join([w for w,_ in top])[:60] or "General"

    def _short_quote_or_paraphrase(self, text: str, limit_words: int = 25) -> str:
        words = text.strip().split()
        if len(words) <= limit_words:
            return '"' + ' '.join(words) + '"'
        # Paraphrase heuristic: take first sentence and compress
        sentence = re.split(r"[\.!?]", text)[0]
        return sentence.strip()

    def _confidence_from_support(self, sources: List[str], sims: List[float]) -> str:
        uniq = len(set(sources))
        avg = sum(sims)/len(sims) if sims else 0.0
        score = uniq + avg
        if score >= 3.5:
            return "High"
        if score >= 2.2:
            return "Medium"
        return "Low"

    def _extract_definitions(self, documents: List[Dict[str, Any]]) -> List[str]:
        defs: List[str] = []
        for d in documents:
            text = d.get('content','')
            matches = re.findall(r"([A-Z][A-Za-z\- ]{2,})\s+is\s+(?:an?|the)\s+([^\.]{5,120})", text)
            for term, desc in matches[:1]:
                defs.append(f"- **{term.strip()}**: {desc.strip()} ({d.get('source','Unknown')}{' p.'+str(d.get('page')) if d.get('page') else ''})")
            if len(defs) >= 6:
                break
        return defs[:6]

    def _extract_quantitative(self, documents: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for d in documents:
            txt = d.get('content','')
            nums = re.findall(r"\b\d+[\d\.%]*\b", txt)
            if nums:
                snippet = self._short_quote_or_paraphrase(txt)
                out.append(f"- {snippet} — {d.get('source','Unknown')}{' p.'+str(d.get('page')) if d.get('page') else ''}")
            if len(out) >= 6:
                break
        return out[:6]
    
    def _format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for LLM input."""
        if not documents:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('source', 'Unknown source')
            content = doc.get('content', '')
            relevance_score = doc.get('relevance_score', 0.0)
            
            formatted_doc = f"""
Document {i} (Source: {source}, Relevance: {relevance_score:.3f}):
{content}
---"""
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    def _create_synthesis_prompt(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> str:
        """Create a comprehensive prompt for research-quality information synthesis."""
        
        formatted_docs = self._format_retrieved_documents(documents)
        
        prompt = f"""You are an expert research analyst conducting a comprehensive literature review and analysis. Your task is to synthesize information from multiple sources to provide a research-quality answer that demonstrates deep analytical thinking.

RESEARCH QUESTION: {query}

SUB-QUESTIONS TO ADDRESS:
{chr(10).join(f"- {sq}" for sq in sub_queries)}

AVAILABLE SOURCES:
{formatted_docs}

ANALYSIS REQUIREMENTS:
1. **Deep Analysis**: Don't just summarize - analyze, interpret, and synthesize information
2. **Critical Thinking**: Evaluate the quality, reliability, and relevance of information
3. **Cross-Source Synthesis**: Compare, contrast, and identify patterns across sources
4. **Insight Generation**: Extract meaningful insights, trends, and implications
5. **Gap Identification**: Clearly identify what information is missing or needs further research
6. **Structured Output**: Present findings in a professional, research-quality format

CRITICAL INSTRUCTIONS:
- NEVER use placeholders like \\1, \\2, or any template variables
- NEVER truncate or cut off your analysis
- ALWAYS provide complete, coherent analysis
- ALWAYS base conclusions on evidence from retrieved documents
- ALWAYS identify specific knowledge gaps
- ALWAYS provide actionable insights
- ALWAYS complete your full analysis without stopping mid-sentence

RESPONSE STRUCTURE:
# Research Report on {query}

## Executive Summary
[2-3 sentence overview of key findings and overall assessment]

## Document Analysis

### Document Overview
- **Document Name**: [Extract from sources if available]
- **Content Type**: [Identify document type and purpose]
- **Key Topics**: [Main themes and subjects covered]

### Key Findings
[For each sub-question, provide specific insights with page references when available]

### [Sub-question 1]
[Deep analysis with specific insights, not just content repetition]
- **Page Reference**: [If available from source]
- **Key Points**: [Specific findings]
- **Evidence**: [Direct quotes or references]

### [Sub-question 2]
[Continue for each sub-question with analytical insights]

## Cross-Source Analysis
- **Agreements**: What do sources agree on?
- **Contradictions**: Where do sources disagree or present conflicting information?
- **Patterns**: What patterns or trends emerge across sources?
- **Quality Assessment**: How reliable and comprehensive are the sources?

## Key Insights and Implications
- **Primary Insights**: What are the most important findings?
- **Trends Identified**: What trends or patterns are evident?
- **Implications**: What do these findings mean for the broader topic?
- **Practical Applications**: How can this information be applied?

## Knowledge Gaps and Limitations
- **Missing Information**: What key information is not available?
- **Source Limitations**: What are the limitations of the current sources?
- **Research Recommendations**: What additional research would be valuable?

## Conclusion
[Overall assessment, key takeaways, and recommendations for further research]

IMPORTANT: Provide genuine analysis and synthesis, not just content repetition. Demonstrate critical thinking and research-level insights that go beyond simple summarization. Ensure your response is complete and never truncated."""

        return prompt
    
    def _synthesize_with_openai(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> str:
        """Synthesize information using OpenAI API."""
        try:
            prompt = self._create_synthesis_prompt(query, documents, sub_queries)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst with deep expertise in information synthesis, critical analysis, and academic research. You excel at identifying patterns, extracting insights, and providing research-quality analysis that goes beyond simple summarization. You are skilled at cross-source analysis, trend identification, and gap analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.2,  # Lower temperature for more focused analysis
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Successfully synthesized answer using OpenAI")
            return answer
            
        except Exception as e:
            logger.error(f"Error in OpenAI synthesis: {e}")
            return self._fallback_synthesis(query, documents, sub_queries)
    
    def _fallback_synthesis(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> str:
        """Advanced fallback synthesis with real research-level analysis."""
        if not documents:
            return self._generate_no_documents_response(query)
        
        # Perform deep analysis
        analysis = self._perform_deep_analysis(query, documents, sub_queries)
        
        # Generate structured research-quality answer
        return self._generate_research_report(query, analysis, documents)
    
    def _generate_no_documents_response(self, query: str) -> str:
        """Generate helpful response when no documents are found."""
        return f"""# Research Analysis: {query}

## Executive Summary
No relevant information was found in the uploaded documents to answer your query about "{query}".

## Recommendations for Better Results

### 1. Query Refinement
- **Be more specific**: Instead of "{query}", try asking about specific aspects
- **Use related terms**: Consider synonyms or related concepts
- **Break down complex queries**: Ask about individual components separately

### 2. Document Enhancement
- **Upload relevant documents**: Ensure your documents contain information about "{query}"
- **Check document quality**: Verify documents are readable and contain relevant content
- **Consider document types**: Different document types may contain different perspectives

### 3. Alternative Approaches
- **Try broader queries**: Ask about related topics that might be covered
- **Use different keywords**: Experiment with alternative terminology
- **Focus on specific aspects**: Break down your question into smaller parts

## Next Steps
1. Review and refine your research question
2. Upload additional relevant documents
3. Try alternative query formulations
4. Consider the scope and focus of your research

*This analysis was performed using advanced document processing and semantic search capabilities.*"""
    
    def _perform_deep_analysis(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> Dict[str, Any]:
        """Perform deep analytical analysis of retrieved documents."""
        analysis = {
            'query_analysis': self._analyze_query_intent(query),
            'document_analysis': self._analyze_documents(documents),
            'sub_query_insights': {},
            'cross_source_analysis': self._perform_cross_source_analysis(documents),
            'trend_analysis': self._identify_trends_and_patterns(documents),
            'implications': self._extract_implications(documents, query),
            'knowledge_gaps': self._identify_knowledge_gaps(documents, sub_queries)
        }
        
        # Analyze each sub-query
        for sub_query in sub_queries:
            analysis['sub_query_insights'][sub_query] = self._analyze_sub_query(sub_query, documents)
        
        return analysis
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and scope of the research query."""
        query_lower = query.lower()
        
        intent_analysis = {
            'primary_intent': 'informational',
            'scope': 'general',
            'complexity': 'medium',
            'expected_answer_type': 'explanatory'
        }
        
        # Determine intent
        if any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            intent_analysis['primary_intent'] = 'definitional'
            intent_analysis['expected_answer_type'] = 'explanatory'
        elif any(word in query_lower for word in ['how', 'process', 'method', 'steps']):
            intent_analysis['primary_intent'] = 'procedural'
            intent_analysis['expected_answer_type'] = 'instructional'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'purpose']):
            intent_analysis['primary_intent'] = 'causal'
            intent_analysis['expected_answer_type'] = 'analytical'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            intent_analysis['primary_intent'] = 'comparative'
            intent_analysis['expected_answer_type'] = 'comparative'
        elif any(word in query_lower for word in ['benefit', 'advantage', 'pros', 'cons']):
            intent_analysis['primary_intent'] = 'evaluative'
            intent_analysis['expected_answer_type'] = 'evaluative'
        
        # Determine scope
        if len(query.split()) <= 3:
            intent_analysis['scope'] = 'broad'
            intent_analysis['complexity'] = 'high'
        elif len(query.split()) <= 8:
            intent_analysis['scope'] = 'focused'
            intent_analysis['complexity'] = 'medium'
        else:
            intent_analysis['scope'] = 'specific'
            intent_analysis['complexity'] = 'low'
        
        return intent_analysis
    
    def _analyze_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the retrieved documents for quality and relevance."""
        if not documents:
            return {'quality': 'none', 'relevance': 'none', 'coverage': 'none'}
        
        # Calculate relevance scores
        relevance_scores = [doc.get('relevance_score', 0) for doc in documents]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Analyze content quality
        content_lengths = [len(doc.get('content', '')) for doc in documents]
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        
        # Determine quality levels
        if avg_relevance > 0.7:
            relevance_level = 'high'
        elif avg_relevance > 0.4:
            relevance_level = 'medium'
        else:
            relevance_level = 'low'
        
        if avg_content_length > 500:
            quality_level = 'high'
        elif avg_content_length > 200:
            quality_level = 'medium'
        else:
            quality_level = 'low'
        
        # Coverage analysis
        unique_sources = len(set(doc.get('source', 'Unknown') for doc in documents))
        if unique_sources >= 3:
            coverage_level = 'comprehensive'
        elif unique_sources >= 2:
            coverage_level = 'moderate'
        else:
            coverage_level = 'limited'
        
        return {
            'quality': quality_level,
            'relevance': relevance_level,
            'coverage': coverage_level,
            'avg_relevance_score': avg_relevance,
            'unique_sources': unique_sources,
            'total_sections': len(documents)
        }
    
    def _perform_cross_source_analysis(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform cross-source analysis to identify patterns and contradictions."""
        if len(documents) < 2:
            return {'consistency': 'single_source', 'contradictions': [], 'agreements': []}
        
        # Group by source
        sources = {}
        for doc in documents:
            source = doc.get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        # Analyze for contradictions and agreements
        contradictions = []
        agreements = []
        
        # Simple keyword-based analysis for contradictions
        all_content = ' '.join([doc.get('content', '').lower() for doc in documents])
        
        # Look for contradictory indicators
        contradiction_indicators = ['however', 'but', 'although', 'despite', 'contrary', 'opposite', 'different']
        for indicator in contradiction_indicators:
            if indicator in all_content:
                contradictions.append(f"Found contradictory language using '{indicator}'")
        
        # Look for agreement indicators
        agreement_indicators = ['similarly', 'likewise', 'also', 'furthermore', 'additionally', 'consistent']
        for indicator in agreement_indicators:
            if indicator in all_content:
                agreements.append(f"Found agreement language using '{indicator}'")
        
        # Determine overall consistency
        if len(contradictions) > len(agreements):
            consistency = 'mixed'
        elif len(agreements) > 0:
            consistency = 'consistent'
        else:
            consistency = 'neutral'
        
        return {
            'consistency': consistency,
            'contradictions': contradictions,
            'agreements': agreements,
            'source_diversity': len(sources)
        }
    
    def _identify_trends_and_patterns(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify trends and patterns across documents."""
        if not documents:
            return {'trends': [], 'patterns': [], 'frequency_analysis': {}}
        
        # Extract key terms and concepts
        all_text = ' '.join([doc.get('content', '') for doc in documents]).lower()
        
        # Simple frequency analysis
        words = all_text.split()
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():  # Only meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent terms
        frequent_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Identify patterns
        patterns = []
        if any('increase' in word or 'growth' in word for word in word_freq.keys()):
            patterns.append("Growth/Increase trend identified")
        if any('decrease' in word or 'decline' in word for word in word_freq.keys()):
            patterns.append("Decline/Decrease trend identified")
        if any('new' in word or 'emerging' in word for word in word_freq.keys()):
            patterns.append("Emerging/New developments identified")
        
        return {
            'trends': patterns,
            'patterns': patterns,
            'frequency_analysis': dict(frequent_terms),
            'key_concepts': [term for term, freq in frequent_terms if freq > 1]
        }
    
    def _extract_implications(self, documents: List[Dict[str, Any]], query: str) -> List[str]:
        """Extract implications and insights from the analysis."""
        implications = []
        
        if not documents:
            return implications
        
        # Analyze document content for implications
        all_content = ' '.join([doc.get('content', '') for doc in documents]).lower()
        
        # Look for implication indicators
        implication_indicators = [
            'implication', 'impact', 'consequence', 'result', 'outcome',
            'significance', 'importance', 'benefit', 'advantage', 'disadvantage'
        ]
        
        for indicator in implication_indicators:
            if indicator in all_content:
                implications.append(f"Document discusses {indicator}s related to the topic")
        
        # Add general implications based on content analysis
        if len(documents) > 3:
            implications.append("Multiple sources provide comprehensive coverage of the topic")
        elif len(documents) == 1:
            implications.append("Limited source coverage - additional sources recommended")
        
        # Add query-specific implications
        if 'technology' in query.lower():
            implications.append("Technology-related implications may include adoption challenges and implementation considerations")
        if 'business' in query.lower():
            implications.append("Business implications may include market impact and competitive advantages")
        
        return implications
    
    def _identify_knowledge_gaps(self, documents: List[Dict[str, Any]], sub_queries: List[str]) -> List[str]:
        """Identify knowledge gaps in the retrieved information."""
        gaps = []
        
        if not documents:
            gaps.append("No relevant documents found for analysis")
            return gaps
        
        # Check coverage of sub-queries
        for sub_query in sub_queries:
            relevant_docs = [doc for doc in documents if self._is_relevant_to_subquery(doc, sub_query)]
            if not relevant_docs:
                gaps.append(f"No information found for: {sub_query}")
        
        # Check for missing perspectives
        sources = set(doc.get('source', 'Unknown') for doc in documents)
        if len(sources) < 2:
            gaps.append("Limited source diversity - only one document source available")
        
        # Check for recent information
        recent_indicators = ['2024', '2023', 'recent', 'latest', 'current', 'new']
        all_content = ' '.join([doc.get('content', '') for doc in documents]).lower()
        if not any(indicator in all_content for indicator in recent_indicators):
            gaps.append("No recent information found - documents may be outdated")
        
        # Check for quantitative data
        if not any(char.isdigit() for char in all_content):
            gaps.append("No quantitative data or statistics found in the documents")
        
        return gaps
    
    def _analyze_sub_query(self, sub_query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a specific sub-query against the documents."""
        relevant_docs = [doc for doc in documents if self._is_relevant_to_subquery(doc, sub_query)]
        
        if not relevant_docs:
            return {
                'coverage': 'none',
                'key_points': [],
                'sources': [],
                'confidence': 'low'
            }
        
        # Extract key points
        key_points = []
        for doc in relevant_docs[:3]:  # Top 3 most relevant
            content = doc.get('content', '')
            # Extract first sentence as key point
            first_sentence = content.split('.')[0] if '.' in content else content[:100]
            if first_sentence.strip():
                key_points.append(first_sentence.strip())
        
        # Determine confidence
        if len(relevant_docs) >= 3:
            confidence = 'high'
        elif len(relevant_docs) >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'coverage': 'good' if relevant_docs else 'none',
            'key_points': key_points,
            'sources': [doc.get('source', 'Unknown') for doc in relevant_docs],
            'confidence': confidence,
            'relevance_scores': [doc.get('relevance_score', 0) for doc in relevant_docs]
        }
    
    def _generate_research_report(self, query: str, analysis: Dict[str, Any], documents: List[Dict[str, Any]]) -> str:
        """Generate a structured research-quality report."""
        report_parts = []
        
        # Header & Executive Summary (document-aware, no boilerplate)
        report_parts.append(f"# Research Report on {query}\n")
        report_parts.append("## Executive Summary")
        doc_analysis = analysis['document_analysis']
        summary_line = (
            f"Analyzed {doc_analysis['total_sections']} sections across {doc_analysis['unique_sources']} sources. "
            f"Overall coverage: {doc_analysis['coverage']}; relevance: {doc_analysis['relevance']}."
        )
        report_parts.append(summary_line + "\n")
        
        # Key Findings
        report_parts.append("## Key Findings")
        
        # Process each sub-query
        for sub_query, insights in analysis['sub_query_insights'].items():
            report_parts.append(f"### {sub_query}")
            
            if insights['coverage'] == 'none':
                report_parts.append("*No specific information found for this aspect.*")
            else:
                report_parts.append(f"**Coverage:** {insights['coverage'].title()} (Confidence: {insights['confidence'].title()})")
                
                if insights['key_points']:
                    report_parts.append("**Key Points:**")
                    for point in insights['key_points']:
                        # Try to hint page reference if available in content markers
                        report_parts.append(f"- {point}")
                
                if insights['sources']:
                    report_parts.append(f"**Sources:** {', '.join(insights['sources'])}")
            
            report_parts.append("")  # Add spacing
        
        # Cross-Source Analysis
        cross_analysis = analysis['cross_source_analysis']
        if cross_analysis['source_diversity'] > 1:
            report_parts.append("## Cross-Source Analysis")
            report_parts.append(f"**Consistency:** {cross_analysis['consistency'].title()}")
            
            if cross_analysis['agreements']:
                report_parts.append("**Agreements Found:**")
                for agreement in cross_analysis['agreements']:
                    report_parts.append(f"- {agreement}")
            
            if cross_analysis['contradictions']:
                report_parts.append("**Contradictions Found:**")
                for contradiction in cross_analysis['contradictions']:
                    report_parts.append(f"- {contradiction}")
            
            report_parts.append("")
        
        # Trends and Patterns
        trend_analysis = analysis['trend_analysis']
        if trend_analysis['trends']:
            report_parts.append("## Trends and Patterns")
            for trend in trend_analysis['trends']:
                report_parts.append(f"- {trend}")
            report_parts.append("")
        
        # Implications
        if analysis['implications']:
            report_parts.append("## Implications and Insights")
            for implication in analysis['implications']:
                report_parts.append(f"- {implication}")
            report_parts.append("")
        
        # Knowledge Gaps (make gaps specific, filter generic phrases)
        if analysis['knowledge_gaps']:
            specific_gaps = [g for g in analysis['knowledge_gaps'] if 'limited' not in g.lower() or len(analysis['knowledge_gaps']) == 1]
            if specific_gaps:
                report_parts.append("## Knowledge Gaps")
                for gap in specific_gaps:
                    report_parts.append(f"- {gap}")
                report_parts.append("")
        
        # Conclusion
        report_parts.append("## Conclusion")
        conclusion = (
            f"Coverage: {doc_analysis['coverage']}; relevance: {doc_analysis['relevance']}. "
        )
        if analysis['knowledge_gaps']:
            conclusion += "Some specific gaps remain; additional sources may be required."
        else:
            conclusion += "Evidence suggests the findings are well-supported by available sources."
        report_parts.append(conclusion)
        report_parts.append("\n\n*Generated locally via document embeddings and semantic retrieval.*")
        
        return "\n".join(report_parts)
    
    def _is_relevant_to_subquery(self, doc: Dict[str, Any], sub_query: str) -> bool:
        """Simple relevance check for fallback synthesis."""
        content = doc.get('content', '').lower()
        sub_query_words = sub_query.lower().split()
        
        # Check if any significant words from sub-query appear in content
        significant_words = [word for word in sub_query_words if len(word) > 3]
        matches = sum(1 for word in significant_words if word in content)
        
        return matches >= len(significant_words) * 0.3  # At least 30% of significant words match
    
    def _extract_key_points(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from retrieved documents."""
        key_points = []
        for doc in documents:
            content = doc.get('content', '')
            source = doc.get('source', 'Unknown')
            page = doc.get('page')
            # Split into sentences and pick 1-2 salient ones
            sentences = [s.strip() for s in re.split(r'[\.!?]\s+', content) if len(s.strip()) > 30]
            for s in sentences[:2]:
                ref = f" (page {page})" if page else ""
                key_points.append(f"- {s}{ref} — {source}")
        return key_points[:10]

    def _group_by_theme(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group document chunks into rough themes using simple keyword buckets."""
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        themes = {
            'Applications': ['apply', 'application', 'use case', 'usage', 'deploy', 'implement'],
            'Challenges': ['challenge', 'limitation', 'issue', 'risk', 'barrier', 'problem'],
            'Ethics': ['ethic', 'privacy', 'bias', 'fairness', 'transparen', 'accountab'],
            'Methodology': ['method', 'approach', 'algorithm', 'model', 'technique', 'framework'],
            'Results': ['result', 'finding', 'evaluation', 'experiment', 'performance'],
        }
        for doc in documents:
            text = doc.get('content', '').lower()
            matched = False
            for theme, kws in themes.items():
                if any(kw in text for kw in kws):
                    buckets.setdefault(theme, []).append(doc)
                    matched = True
                    break
            if not matched:
                buckets.setdefault('General', []).append(doc)
        return buckets

    def _build_report(self, title: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> str:
        """Research-grade structured Markdown report with provenance, citations, and confidence."""
        if not documents:
            return f"# Research Report on {title}\n\nInsufficient evidence — consider uploading additional sources or refining your query."

        # Filter low-quality by similarity threshold
        keep_docs = [d for d in documents if d.get('relevance_score', 0.0) >= 0.05]
        excluded = [d for d in documents if d.get('relevance_score', 0.0) < 0.05]

        # Clustering into themes
        clusters = self._cluster_chunks(keep_docs, similarity_threshold=0.60)
        labeled_clusters: List[Tuple[str,List[int]]] = []
        for idxs in clusters:
            label = self._label_cluster(keep_docs, idxs)
            labeled_clusters.append((label, idxs))

        # Executive Summary
        key_points = self._extract_key_points(keep_docs)
        tldr = " ".join([kp.split('—')[0].lstrip('- ').strip() for kp in key_points[:3]])
        if not tldr:
            tldr = "This report synthesizes the most relevant evidence from the uploaded documents."

        # Stats
        doc_ids = list({d.get('doc_id','unknown') for d in keep_docs})
        filenames = list({d.get('source','Unknown') for d in keep_docs})
        section_count = len(keep_docs)
        doc_count = len(set(filenames))

        parts: List[str] = []
        parts.append(f"# Research Report on {title} (docs: {doc_count}, sections: {section_count})")
        parts.append("\n## TL;DR\n")
        parts.append(tldr + "\n")
        parts.append("\n## Executive Summary\n")
        parts.append("This report integrates evidence across retrieved sections to answer the question with explicit citations and confidence estimates.\n")

        # Definitions
        definitions = self._extract_definitions(keep_docs)
        if definitions:
            parts.append("\n## Definitions / Key Concepts\n")
            parts.extend(definitions)

        # Method / Evidence Overview
        threshold = 0.05
        parts.append("\n## Method / Evidence Overview\n")
        parts.append(f"- Considered {section_count} sections from {doc_count} documents.\n- Similarity threshold: {threshold}.\n- Excluded {len(excluded)} low-similarity sections.")

        # Detailed Findings by theme
        parts.append("\n## Detailed Findings\n")
        for label, idxs in labeled_clusters:
            parts.append(f"### {label[:80]}\n")
            # Build 3–6 findings
            group_docs = [keep_docs[i] for i in idxs]
            bullets = 0
            for gd in group_docs:
                if bullets >= 6:
                    break
                snippet = self._short_quote_or_paraphrase(gd.get('content',''))
                src = gd.get('source','Unknown')
                page = f", p.{gd.get('page')}" if gd.get('page') else ""
                sim = gd.get('relevance_score',0.0)
                parts.append(f"- {snippet} **[{src}{page}]** (sim={sim:.2f})")
                bullets += 1

            # Confidence for the theme
            theme_conf = self._confidence_from_support([d.get('source','Unknown') for d in group_docs], [d.get('relevance_score',0.0) for d in group_docs])
            parts.append(f"- Confidence for this theme: **{theme_conf}**\n")

            # Provenance footnote
            provin = "; ".join([f"{d.get('source','Unknown')}{' p.'+str(d.get('page')) if d.get('page') else ''} (sim={d.get('relevance_score',0.0):.2f})" for d in group_docs[:6]])
            parts.append(f"_Provenance: {provin}_\n")

        # Cross-Document Analysis
        parts.append("\n## Cross-Document Analysis\n")
        # Simple agreements/contradictions via keywords
        all_text = "\n".join(d.get('content','').lower() for d in keep_docs)
        agreements = ["multiple sources make similar claims", "consistent terminology across documents"] if any(w in all_text for w in ["consisten","similar"]) else []
        contradictions = ["contradictory statements present"] if any(w in all_text for w in ["however","contrary","but "]) else []
        if agreements:
            parts.append("- Agreements: " + "; ".join(agreements))
        if contradictions:
            parts.append("- Contradictions: " + "; ".join(contradictions))

        # Quantitative / Practical Details
        quant = self._extract_quantitative(keep_docs)
        if quant:
            parts.append("\n## Quantitative / Practical Details\n")
            parts.extend(quant)

        # Knowledge Gaps
        gaps: List[str] = []
        if sub_queries:
            all_lower = "\n".join(d.get('content','').lower() for d in keep_docs)
            for sq in sub_queries:
                sq_terms = [t for t in sq.lower().split() if len(t) > 3]
                if not any(term in all_lower for term in sq_terms):
                    gaps.append(f"- Missing coverage for: **{sq}**. Suggest: refine keywords or upload documents addressing this aspect.")
        if gaps:
            parts.append("\n## Knowledge Gaps\n")
            parts.extend(gaps)

        # Implications & Recommendations
        parts.append("\n## Implications & Recommendations\n")
        parts.append("- Prioritize additional evidence for weakly supported themes.\n- Triangulate findings across independent sources.\n- For insufficient coverage, consider targeted searches with domain-specific keywords.\n- If metrics are needed, seek benchmark reports or empirical studies." )

        # Sources / Provenance
        parts.append("\n## Sources / Provenance\n")
        unique_sources = {}
        for d in keep_docs:
            key = (d.get('source','Unknown'), d.get('doc_id',''))
            unique_sources.setdefault(key, []).append(d.get('page'))
        i = 1
        for (src, did), pages in unique_sources.items():
            pages_list = sorted([p for p in pages if p is not None])
            page_str = f" pages {pages_list}" if pages_list else ""
            parts.append(f"{i}. {src} (doc_id={did}){page_str}")
            i += 1

        # Actionable summary
        parts.append("\n## Actionable Summary\n")
        parts.append("This report consolidates the strongest available evidence and highlights gaps requiring targeted data collection or further reading.")

        return "\n".join(parts)
    
    def _add_citations(self, answer: str, documents: List[Dict[str, Any]]) -> str:
        """Add citation information to the answer."""
        if not config.INCLUDE_CITATIONS or not documents:
            return answer
        
        # Extract unique sources
        sources = list(set(doc.get('source', 'Unknown') for doc in documents))
        
        # Add citations section
        citations_section = "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            citations_section += f"{i}. {source}\n"
        
        return answer + citations_section
    
    def synthesize(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> str:
        """
        Synthesize information from retrieved documents into a comprehensive answer.
        
        Args:
            query: Original research query
            documents: List of retrieved document chunks
            sub_queries: List of sub-questions derived from the main query
            
        Returns:
            Synthesized answer string
        """
        try:
            logger.info(f"Starting synthesis for query: {query[:50]}... with {len(documents)} documents")
            # Log doc_ids being used for synthesis for debugging scope
            try:
                used_docs = list({d.get('doc_id','unknown') for d in documents})
                logger.info(f"[Synthesis Scope] doc_ids used: {used_docs}")
            except Exception:
                pass
            
            if not documents:
                return f"I couldn't find any relevant information to answer '{query}'. Please upload relevant documents or try rephrasing your question."
            
            # Choose synthesis method
            logger.info(f"Synthesis backend: {self.backend}; chunks passed: {len(documents)}")
            if self.backend == "openai" and self.client is not None:
                prompt = self._build_llm_prompt(query, documents, sub_queries)
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role":"system","content":"You are a research synthesis assistant. Produce a deep, evidence-based, well-structured Markdown report with citations and a reasoning trace."},
                            {"role":"user","content":prompt}
                        ],
                        temperature=0.2,
                        max_tokens=3000
                    )
                    answer = resp.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"OpenAI synthesis failed: {e}")
                    answer = self._build_report(query, documents, sub_queries)
            elif self.backend == "huggingface" and hasattr(self, "hf_gen"):
                prompt = self._build_llm_prompt(query, documents, sub_queries)
                try:
                    out = self.hf_gen(prompt, max_new_tokens=1200, do_sample=False)
                    answer = out[0]["generated_text"][len(prompt):].strip()
                except Exception as e:
                    logger.error(f"HF synthesis failed: {e}")
                    answer = self._build_report(query, documents, sub_queries)
            else:
                answer = self._build_report(query, documents, sub_queries)
            
            # Add citations if enabled
            if config.INCLUDE_CITATIONS:
                answer = self._add_citations(answer, documents)
            
            # Ensure answer is complete and not truncated
            if len(answer) > config.MAX_ANSWER_LENGTH:
                logger.warning(f"Answer length ({len(answer)}) exceeds limit ({config.MAX_ANSWER_LENGTH}). Consider increasing MAX_ANSWER_LENGTH.")
                # Don't truncate - let the full answer through for research quality
            
            logger.info(f"Synthesis completed. Answer length: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return f"I encountered an error while synthesizing the information for '{query}'. Please try again or contact support if the issue persists."

    def _build_llm_prompt(self, query: str, documents: List[Dict[str, Any]], sub_queries: List[str]) -> str:
        # Compose retrieved chunks block
        lines = []
        for i, d in enumerate(documents[:30], 1):
            src = d.get('source','Unknown')
            page = d.get('page')
            sim = d.get('relevance_score',0.0)
            content = d.get('content','')
            lines.append(f"[Chunk {i}] Source: {src}{' p.'+str(page) if page else ''} | sim={sim:.2f}\n{content}\n---")
        chunks_block = "\n".join(lines)

        subq = "\n".join(f"- {sq}" for sq in (sub_queries or [query]))
        prompt = f"""
Task: Synthesize a comprehensive research answer in well-structured Markdown for the following query.

Query:
{query}

Sub-questions:
{subq}

Retrieved Evidence (document-aware; cite filename and page):
{chunks_block}

Instructions:
- Produce: Title, Executive Summary (3–5 sentences), Definitions/Key Concepts, Method/Evidence Overview, Detailed Findings grouped by theme (3–6 bullets each with short explanation + citation + confidence), Cross-Document Analysis (agreements/contradictions/unique perspectives), Quantitative/Practical Details, Knowledge Gaps (what’s missing, why it matters, suggested actions), Implications & Recommendations (3–5 next steps), Sources/Provenance (numbered list with pages used), Actionable Summary.
- Keep quotes ≤ 25 words; otherwise paraphrase.
- No placeholders. If evidence is missing for a claim, mark as Not supported by available documents and add to Knowledge Gaps.
- Return Markdown only.
"""
        return prompt
    
    def get_synthesis_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the synthesis process."""
        if not documents:
            return {"document_count": 0, "source_count": 0, "total_content_length": 0}
        
        sources = set(doc.get('source', 'Unknown') for doc in documents)
        total_length = sum(len(doc.get('content', '')) for doc in documents)
        avg_relevance = sum(doc.get('relevance_score', 0) for doc in documents) / len(documents)
        
        return {
            "document_count": len(documents),
            "source_count": len(sources),
            "total_content_length": total_length,
            "average_relevance_score": avg_relevance,
            "sources": list(sources)
        }
