import React from 'react';
import { FileText, Search, Brain, AlertTriangle, CheckCircle, Download, Copy, Share2 } from 'lucide-react';

interface DocumentResearchProps {
  documentResearchResponse: {
    answer: string;
    reasoning_trace: string[];
    gaps: string[];
    document_info?: {
      id: string;
      filename: string;
      text_length: number;
      chunk_count: number;
      added_at: string;
    };
    research_questions?: string[];
    insights?: Array<{
      type: string;
      title: string;
      content: string;
      confidence: string;
      sources: string[];
    }>;
    retrieved_sections?: number;
    session_id?: string;
  };
  onExport: (format: 'pdf' | 'markdown') => void;
}

const DocumentResearch: React.FC<DocumentResearchProps> = ({ documentResearchResponse, onExport }) => {
  const { answer, document_info, research_questions, insights, reasoning_trace, gaps } = documentResearchResponse;

  const handleCopyInsights = () => {
    if (insights && insights.length > 0) {
      const insightsText = insights.map(insight => `${insight.title}\n${insight.content}`).join('\n\n');
      navigator.clipboard.writeText(insightsText);
    } else {
      navigator.clipboard.writeText(answer);
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence.toLowerCase()) {
      case 'high': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'document overview': return FileText;
      case 'content analysis': return Brain;
      case 'research finding': return Search;
      default: return CheckCircle;
    }
  };

  return (
    <div className="space-y-8">
      {/* Document Information */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
            <FileText className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Document Analysis Complete</h2>
            <p className="text-sm text-gray-600">Comprehensive research on uploaded document</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-500">Document</p>
            <p className="font-semibold text-gray-900">{document_info?.filename || 'Unknown'}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-500">Content Size</p>
            <p className="font-semibold text-gray-900">{document_info?.text_length?.toLocaleString() || '0'} characters</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-500">Sections Analyzed</p>
            <p className="font-semibold text-gray-900">{document_info?.chunk_count || 0} chunks</p>
          </div>
        </div>
      </div>

      {/* Research Questions */}
      {research_questions && research_questions.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Brain className="w-6 h-6 text-purple-600" />
            <h3 className="text-lg font-semibold text-gray-900">Research Questions Generated</h3>
          </div>
          <div className="space-y-3">
            {research_questions.map((question, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg">
                <span className="flex-shrink-0 w-6 h-6 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                  {index + 1}
                </span>
                <p className="text-gray-800">{question}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Answer */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Search className="w-6 h-6 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Research Analysis</h3>
        </div>
        <div className="prose prose-gray max-w-none">
          <div className="whitespace-pre-wrap text-gray-800 leading-relaxed">
            {answer}
          </div>
        </div>
      </div>

      {/* Document Insights */}
      <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-xl shadow-lg border-2 border-green-200 p-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Document Insights</h2>
              <p className="text-sm text-gray-600">Comprehensive analysis results</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={handleCopyInsights}
              className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-blue-600 hover:bg-white/50 rounded-lg transition-all duration-200"
            >
              <Copy className="w-4 h-4" />
              <span className="text-sm font-medium">Copy</span>
            </button>
            <button
              onClick={() => onExport('markdown')}
              className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-blue-600 hover:bg-white/50 rounded-lg transition-all duration-200"
            >
              <FileText className="w-4 h-4" />
              <span className="text-sm font-medium">Export MD</span>
            </button>
            <button
              onClick={() => onExport('pdf')}
              className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-105"
            >
              <Download className="w-4 h-4" />
              <span className="font-medium">Export PDF</span>
            </button>
          </div>
        </div>

        <div className="space-y-6">
          {insights && insights.length > 0 ? insights.map((insight, index) => {
            const IconComponent = getInsightIcon(insight.type);
            return (
              <div key={index} className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                      <IconComponent className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900">{insight.title}</h4>
                      <p className="text-sm text-gray-500">{insight.type}</p>
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${getConfidenceColor(insight.confidence)}`}>
                    {insight.confidence} Confidence
                  </span>
                </div>
                <p className="text-gray-800 leading-relaxed mb-4">{insight.content}</p>
                {insight.sources.length > 0 && (
                  <div className="pt-4 border-t border-gray-100">
                    <p className="text-sm text-gray-500 mb-2">Sources:</p>
                    <div className="flex flex-wrap gap-2">
                      {insight.sources.map((source, idx) => (
                        <span key={idx} className="inline-flex items-center px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                          {source}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          }) : (
            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 text-center">
              <p className="text-gray-500">No detailed insights available. The main analysis is shown above.</p>
            </div>
          )}
        </div>
      </div>

      {/* Knowledge Gaps */}
      {gaps.length > 0 && (
        <div className="bg-amber-50 border-2 border-amber-200 rounded-xl p-6">
          <div className="flex items-center space-x-2 mb-4">
            <AlertTriangle className="w-6 h-6 text-amber-600" />
            <h3 className="text-lg font-semibold text-amber-900">Knowledge Gaps Identified</h3>
          </div>
          
          <div className="space-y-3">
            {gaps.map((gap, index) => (
              <div key={index} className="flex items-start space-x-3">
                <span className="flex-shrink-0 w-5 h-5 bg-amber-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">
                  {index + 1}
                </span>
                <p className="text-amber-800 text-sm">{gap}</p>
              </div>
            ))}
          </div>
          
          <div className="mt-4 p-3 bg-amber-100 rounded-lg">
            <p className="text-xs text-amber-700">
              <strong>Note:</strong> These gaps indicate areas where additional information or analysis 
              might be needed for a more comprehensive understanding of the document.
            </p>
          </div>
        </div>
      )}

      {/* Success Footer */}
      <div className="bg-green-50 border border-green-200 rounded-xl p-6">
        <div className="flex items-center justify-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Document analysis completed successfully</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse delay-75"></div>
            <span className="text-sm text-gray-600">Full transparency provided</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse delay-150"></div>
            <span className="text-sm text-gray-600">Ready for export</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentResearch;
