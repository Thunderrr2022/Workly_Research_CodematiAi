"""
ReportExporter - Dynamic report generation system
Generates comprehensive PDF and Markdown reports from research results.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import markdown
from jinja2 import Template

from config import config

logger = logging.getLogger(__name__)

class ReportExporter:
    """
    Comprehensive report exporter that generates professional PDF and Markdown reports
    from research results, reasoning traces, and knowledge gaps.
    """
    
    def __init__(self):
        """Initialize the ReportExporter."""
        self.output_dir = config.REPORT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize styles for PDF generation
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        logger.info("ReportExporter initialized")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for PDF generation."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Gap text style
        self.styles.add(ParagraphStyle(
            name='GapText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            leftIndent=20,
            textColor=colors.darkred
        ))
        
        # Reasoning step style
        self.styles.add(ParagraphStyle(
            name='ReasoningStep',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            leftIndent=20,
            textColor=colors.darkblue
        ))
    
    def _create_pdf_metadata(self, query: str) -> Dict[str, str]:
        """Create metadata for PDF document."""
        return {
            'title': f'Research Report: {query[:50]}...',
            'author': 'Deep Researcher Agent',
            'subject': 'AI-Powered Research Analysis',
            'keywords': 'research, analysis, AI, knowledge synthesis',
            'creator': 'Deep Researcher Agent v1.0'
        }
    
    def _add_header_footer(self, canvas, doc):
        """Add header and footer to PDF pages."""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.darkblue)
        canvas.drawString(50, A4[1] - 50, "Deep Researcher Agent - Research Report")
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(50, 50, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(A4[0] - 50, 50, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _create_reasoning_trace_table(self, reasoning_trace: List[str]) -> Table:
        """Create a table for reasoning trace steps."""
        if not reasoning_trace:
            return Paragraph("No reasoning trace available.", self.styles['CustomBodyText'])
        
        # Prepare table data
        table_data = [['Step', 'Description']]
        for i, step in enumerate(reasoning_trace, 1):
            # Extract step description (remove "Step X:" prefix if present)
            description = step
            if step.startswith(f"Step {i}:"):
                description = step[len(f"Step {i}:"):].strip()
            
            table_data.append([str(i), description])
        
        # Create table
        table = Table(table_data, colWidths=[0.5*inch, 6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        return table
    
    def _create_gaps_table(self, gaps: List[str]) -> Table:
        """Create a table for knowledge gaps."""
        if not gaps:
            return Paragraph("No knowledge gaps identified.", self.styles['CustomBodyText'])
        
        # Prepare table data
        table_data = [['Gap #', 'Description']]
        for i, gap in enumerate(gaps, 1):
            table_data.append([str(i), gap])
        
        # Create table
        table = Table(table_data, colWidths=[0.5*inch, 6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        return table
    
    def _create_documents_table(self, documents: List[str]) -> Table:
        """Create a table for analyzed documents."""
        if not documents:
            return Paragraph("No documents were analyzed.", self.styles['CustomBodyText'])
        
        # Prepare table data
        table_data = [['Document #', 'Filename']]
        for i, doc in enumerate(documents, 1):
            table_data.append([str(i), doc])
        
        # Create table
        table = Table(table_data, colWidths=[0.5*inch, 6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        return table
    
    def export_pdf(self, query: str, answer: str, reasoning_trace: List[str], 
                   gaps: List[str], documents: List[str], research_type: str = "query") -> str:
        """
        Export research report as PDF.
        
        Args:
            query: Original research query or document name
            answer: Synthesized answer or insights
            reasoning_trace: List of reasoning steps
            gaps: List of knowledge gaps
            documents: List of analyzed documents
            research_type: Type of research ("query" or "document")
            
        Returns:
            Path to generated PDF file
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_query = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"research_report_{safe_query}_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build content
            story = []
            
            # Title
            story.append(Paragraph("Deep Researcher Agent", self.styles['CustomTitle']))
            if research_type == "document":
                story.append(Paragraph("Document Research Report", self.styles['CustomTitle']))
            else:
                story.append(Paragraph("Query Research Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Metadata
            story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['CustomBodyText']))
            if research_type == "document":
                story.append(Paragraph(f"<b>Document:</b> {query}", self.styles['CustomBodyText']))
            else:
                story.append(Paragraph(f"<b>Query:</b> {query}", self.styles['CustomBodyText']))
            story.append(Spacer(1, 20))
            
            # Research Subject
            if research_type == "document":
                story.append(Paragraph("Document Analysis", self.styles['SectionHeader']))
                story.append(Paragraph(f"Comprehensive analysis of: {query}", self.styles['CustomBodyText']))
            else:
                story.append(Paragraph("Research Query", self.styles['SectionHeader']))
                story.append(Paragraph(query, self.styles['CustomBodyText']))
            story.append(Spacer(1, 20))
            
            # Analyzed Documents
            story.append(Paragraph("Analyzed Documents", self.styles['SectionHeader']))
            story.append(self._create_documents_table(documents))
            story.append(Spacer(1, 20))
            
            # Reasoning Process
            story.append(Paragraph("Research Process", self.styles['SectionHeader']))
            story.append(Paragraph("The following steps were taken to analyze your query:", self.styles['CustomBodyText']))
            story.append(Spacer(1, 10))
            story.append(self._create_reasoning_trace_table(reasoning_trace))
            story.append(Spacer(1, 20))
            
            # Synthesized Answer or Insights
            if research_type == "document":
                story.append(Paragraph("Document Insights", self.styles['SectionHeader']))
            else:
                story.append(Paragraph("Synthesized Answer", self.styles['SectionHeader']))
            # Split answer into paragraphs for better formatting
            answer_paragraphs = answer.split('\n\n')
            for para in answer_paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), self.styles['CustomBodyText']))
                    story.append(Spacer(1, 6))
            story.append(Spacer(1, 20))
            
            # Knowledge Gaps
            if gaps:
                story.append(Paragraph("Knowledge Gaps", self.styles['SectionHeader']))
                story.append(Paragraph("The following knowledge gaps were identified during the research:", self.styles['CustomBodyText']))
                story.append(Spacer(1, 10))
                story.append(self._create_gaps_table(gaps))
                story.append(Spacer(1, 20))
            
            # Footer
            story.append(Paragraph("---", self.styles['CustomBodyText']))
            story.append(Paragraph("This report was generated by the Deep Researcher Agent, an AI-powered multi-agent research system.", self.styles['CustomBodyText']))
            
            # Build PDF
            doc.build(story, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
            
            logger.info(f"PDF report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise
    
    def _create_markdown_template(self) -> Template:
        """Create Jinja2 template for Markdown reports."""
        template_str = """# Deep Researcher Agent - Research Report

**Generated:** {{ timestamp }}  
**Query:** {{ query }}

---

## Research Query

{{ query }}

---

## Analyzed Documents

{% if documents %}
{% for doc in documents %}
- {{ doc }}
{% endfor %}
{% else %}
No documents were analyzed.
{% endif %}

---

## Research Process

The following steps were taken to analyze your query:

{% for step in reasoning_trace %}
{{ loop.index }}. {{ step }}
{% endfor %}

---

## Synthesized Answer

{{ answer }}

---

{% if gaps %}
## Knowledge Gaps

The following knowledge gaps were identified during the research:

{% for gap in gaps %}
{{ loop.index }}. {{ gap }}
{% endfor %}

---

{% endif %}
## Report Information

- **Generated by:** Deep Researcher Agent v1.0
- **Report Type:** {{ report_type }}
- **Generation Time:** {{ timestamp }}
- **Total Reasoning Steps:** {{ reasoning_trace|length }}
- **Knowledge Gaps Identified:** {{ gaps|length }}

---

*This report was generated by the Deep Researcher Agent, an AI-powered multi-agent research system with full chain-of-thought transparency.*
"""
        return Template(template_str)
    
    def export_markdown(self, query: str, answer: str, reasoning_trace: List[str], 
                       gaps: List[str], documents: List[str], research_type: str = "query") -> str:
        """
        Export research report as Markdown.
        
        Args:
            query: Original research query
            answer: Synthesized answer
            reasoning_trace: List of reasoning steps
            gaps: List of knowledge gaps
            documents: List of analyzed documents
            
        Returns:
            Path to generated Markdown file
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_query = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"research_report_{safe_query}_{timestamp}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create template
            template = self._create_markdown_template()
            
            # Render template
            content = template.render(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                query=query,
                documents=documents,
                reasoning_trace=reasoning_trace,
                answer=answer,
                gaps=gaps,
                report_type="Markdown"
            )
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Markdown report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating Markdown report: {e}")
            raise
    
    def get_report_stats(self, query: str, answer: str, reasoning_trace: List[str], 
                        gaps: List[str], documents: List[str]) -> Dict[str, Any]:
        """Get statistics about the report generation."""
        return {
            "query": query,
            "query_length": len(query),
            "answer_length": len(answer),
            "reasoning_steps": len(reasoning_trace),
            "knowledge_gaps": len(gaps),
            "documents_analyzed": len(documents),
            "report_timestamp": datetime.now().isoformat(),
            "estimated_reading_time": max(1, len(answer.split()) // 200)  # Assuming 200 WPM
        }
    
    def list_generated_reports(self) -> List[Dict[str, Any]]:
        """List all generated reports in the output directory."""
        reports = []
        
        try:
            for filename in os.listdir(self.output_dir):
                if filename.endswith(('.pdf', '.md')):
                    filepath = os.path.join(self.output_dir, filename)
                    stat = os.stat(filepath)
                    
                    reports.append({
                        "filename": filename,
                        "filepath": filepath,
                        "format": "PDF" if filename.endswith('.pdf') else "Markdown",
                        "size_bytes": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # Sort by creation time (newest first)
            reports.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing reports: {e}")
        
        return reports
