import React, { useState } from 'react';
import { Search, FileText, Download, CheckCircle } from 'lucide-react';
import QueryInput from './components/QueryInput';
import DocumentUpload from './components/DocumentUpload';
import ReasoningTrace from './components/ReasoningTrace';
import GapsDisplay from './components/GapsDisplay';
import AnswerDisplay from './components/AnswerDisplay';
import DocumentResearch from './components/DocumentResearch';
import LoadingState from './components/LoadingState';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
}

interface QueryResponse {
  answer: string;
  reasoning_trace: string[];
  gaps: string[];
}

interface DocumentResearchResponse {
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
}

function App() {
  const [query, setQuery] = useState('');
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [documentResearchResponse, setDocumentResearchResponse] = useState<DocumentResearchResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [researchMode, setResearchMode] = useState<'query' | 'document'>('query');
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async (userQuery: string) => {
    setQuery(userQuery);
    setIsLoading(true);
    setResponse(null);
    setDocumentResearchResponse(null); // Clear document research response
    setError(null); // Clear any previous errors
    setCurrentStep(0);
    setResearchMode('query');

    try {
      // Simulate progressive loading for better UX
      const steps = [
        'Breaking down your query into sub-questions...',
        'Retrieving relevant documents...',
        'Analyzing and synthesizing information...'
      ];

      for (let i = 0; i < steps.length; i++) {
        setCurrentStep(i + 1);
        await new Promise(resolve => setTimeout(resolve, 1500));
      }

      // Call the actual backend API
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userQuery,
          documents: documents.map(doc => doc.id)
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: QueryResponse = await response.json();
      setResponse(data);
    } catch (error) {
      console.error('Error querying research agent:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setError(`Query research failed: ${errorMessage}. Please ensure the backend server is running on http://localhost:8000.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentResearch = async (file: File) => {
    setIsLoading(true);
    setDocumentResearchResponse(null);
    setResponse(null); // Clear query response
    setError(null); // Clear any previous errors
    setCurrentStep(0);
    setResearchMode('document');

    try {
      // Simulate progressive loading for better UX
      const steps = [
        'Uploading and processing document...',
        'Generating embeddings and analyzing structure...',
        'Extracting insights and key findings...',
        'Identifying knowledge gaps...'
      ];

      for (let i = 0; i < steps.length; i++) {
        setCurrentStep(i + 1);
        await new Promise(resolve => setTimeout(resolve, 1500));
      }

      // Create FormData for file upload
      const formData = new FormData();
      formData.append('files', file);

      // Call the document research upload API
      const response = await fetch(`${API_BASE}/research/document/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data: DocumentResearchResponse = await response.json();
      setDocumentResearchResponse(data);
    } catch (error) {
      console.error('Error researching document:', error);
      
      // Create a more user-friendly error message
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      
      setError(`Document research failed: ${errorMessage}. Please check the file format and ensure the backend server is running.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = async (format: 'pdf' | 'markdown') => {
    const currentResponse = response || documentResearchResponse;
    if (!currentResponse) {
      console.error('No response data available for export');
      return;
    }

    try {
      let exportData;
      if (response) {
        // Query research export
        exportData = {
          query,
          answer: response.answer,
          reasoning_trace: response.reasoning_trace,
          gaps: response.gaps,
          documents: documents.map(doc => doc.name),
          format,
          research_type: 'query'
        };
      } else if (documentResearchResponse) {
        // Document research export
        exportData = {
          query: documentResearchResponse.document_info?.filename || 'Document Analysis',
          answer: documentResearchResponse.answer,
          reasoning_trace: documentResearchResponse.reasoning_trace,
          gaps: documentResearchResponse.gaps,
          documents: [documentResearchResponse.document_info?.filename || 'Unknown Document'],
          format,
          research_type: 'document'
        };
      }

      const exportResponse = await fetch(`${API_BASE}/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData),
      });

      if (!exportResponse.ok) {
        throw new Error(`HTTP error! status: ${exportResponse.status}`);
      }

      // Handle file download
      const blob = await exportResponse.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `research_report_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting report:', error);
      // Fallback: create a simple text file
      let exportData;
      let content;
      
      if (response) {
        exportData = {
          query,
          reasoning_trace: response.reasoning_trace,
          gaps: response.gaps,
          answer: response.answer,
          documents: documents.map(doc => doc.name),
          timestamp: new Date().toISOString()
        };
        
        content = format === 'markdown' 
          ? `# Query Research Report\n\n**Query:** ${query}\n\n**Answer:**\n${response.answer}\n\n**Reasoning Trace:**\n${response.reasoning_trace.map((step, i) => `${i + 1}. ${step}`).join('\n')}\n\n**Knowledge Gaps:**\n${response.gaps.map((gap, i) => `${i + 1}. ${gap}`).join('\n')}`
          : JSON.stringify(exportData, null, 2);
      } else if (documentResearchResponse) {
        const insightsText = documentResearchResponse.insights
          .map(insight => `**${insight.title}**\n${insight.content}`)
          .join('\n\n');
        
        exportData = {
          document: documentResearchResponse.document_info.filename,
          insights: documentResearchResponse.insights,
          reasoning_trace: documentResearchResponse.reasoning_trace,
          gaps: documentResearchResponse.gaps,
          timestamp: new Date().toISOString()
        };
        
        content = format === 'markdown' 
          ? `# Document Research Report\n\n**Document:** ${documentResearchResponse.document_info.filename}\n\n**Insights:**\n${insightsText}\n\n**Reasoning Trace:**\n${documentResearchResponse.reasoning_trace.map((step, i) => `${i + 1}. ${step}`).join('\n')}\n\n**Knowledge Gaps:**\n${documentResearchResponse.gaps.map((gap, i) => `${i + 1}. ${gap}`).join('\n')}`
          : JSON.stringify(exportData, null, 2);
      }
      
      const blob = new Blob([content], { type: format === 'markdown' ? 'text/markdown' : 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `research_report_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Search className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-2xl font-bold">
                <span className="text-blue-600">Workly</span>
                <span className="text-gray-900"> Research</span>
              </h1>
            </div>
          </div>
        </div>
      </header>

      {/* Error Display */}
      {error && (
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <div className="mt-2 text-sm text-red-700">
                  <p>{error}</p>
                </div>
                <div className="mt-4">
                  <button
                    onClick={() => setError(null)}
                    className="bg-red-100 px-3 py-2 rounded-md text-sm font-medium text-red-800 hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        {!response && !isLoading ? (
          <div className="text-center py-12">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                Deep Research Agent
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Multi-agent research system with full chain-of-thought transparency.
                Upload documents and get comprehensive answers with step-by-step reasoning.
              </p>
              
              <DocumentUpload 
          onDocumentsChange={setDocuments} 
          onDocumentResearch={handleDocumentResearch}
        />
              <QueryInput onQuery={handleQuery} hasDocuments={documents.length > 0} />
              
              <div className="flex items-center justify-center space-x-8 mt-12">
                <div className="flex items-center space-x-2 text-gray-600">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Multi-agent system</span>
                </div>
                <div className="flex items-center space-x-2 text-gray-600">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Full transparency</span>
                </div>
                <div className="flex items-center space-x-2 text-gray-600">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Gap detection</span>
                </div>
                <div className="flex items-center space-x-2 text-gray-600">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Document analysis</span>
                </div>
              </div>
            </div>
          </div>
        ) : isLoading ? (
          <LoadingState currentStep={currentStep} query={query} />
        ) : response ? (
          <div className="space-y-8">
            {/* Query Summary */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-2">Research Query</h2>
              <p className="text-gray-600 text-lg">{query}</p>
              {documents.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <p className="text-sm text-gray-500 mb-2">Documents analyzed:</p>
                  <div className="flex flex-wrap gap-2">
                    {documents.map((doc) => (
                      <span key={doc.id} className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                        <FileText className="w-3 h-3 mr-1" />
                        {doc.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Reasoning Trace */}
            <ReasoningTrace steps={response.reasoning_trace} />

            {/* Knowledge Gaps */}
            {response.gaps.length > 0 && (
              <GapsDisplay gaps={response.gaps} />
            )}

            {/* Final Answer */}
            <AnswerDisplay answer={response.answer} onExport={handleExport} />

            {/* New Research Button */}
            <div className="text-center pt-8">
              <button
                onClick={() => {
                  setResponse(null);
                  setDocumentResearchResponse(null);
                  setError(null);
                  setQuery('');
                  setDocuments([]);
                  setResearchMode('query');
                }}
                className="bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Start New Research
              </button>
            </div>
          </div>
        ) : documentResearchResponse ? (
          <div className="space-y-8">
            <DocumentResearch 
              documentResearchResponse={documentResearchResponse}
              onExport={handleExport}
            />
            
            {/* New Research Button */}
            <div className="text-center pt-8">
              <button
                onClick={() => {
                  setResponse(null);
                  setDocumentResearchResponse(null);
                  setError(null);
                  setQuery('');
                  setDocuments([]);
                  setResearchMode('query');
                }}
                className="bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Start New Research
              </button>
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}

export default App;