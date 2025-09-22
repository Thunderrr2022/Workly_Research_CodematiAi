import React from 'react';
import { FileText, Download, CheckCircle, Sparkles, Copy, Share2 } from 'lucide-react';

interface AnswerDisplayProps {
  answer: string;
  onExport: (format: 'pdf' | 'markdown') => void;
}

const AnswerDisplay: React.FC<AnswerDisplayProps> = ({ answer, onExport }) => {
  const handleCopyAnswer = () => {
    navigator.clipboard.writeText(answer);
    // You could add a toast notification here
  };

  return (
    <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-xl shadow-lg border-2 border-green-200 p-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
            <CheckCircle className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Research Complete</h2>
            <p className="text-sm text-gray-600">Synthesized from multiple sources</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={handleCopyAnswer}
            className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-blue-600 hover:bg-white/50 rounded-lg transition-all duration-200"
          >
            <Copy className="w-4 h-4" />
            <span className="text-sm font-medium">Copy</span>
          </button>
          <button
            onClick={() => {/* Add share functionality */}}
            className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-blue-600 hover:bg-white/50 rounded-lg transition-all duration-200"
          >
            <Share2 className="w-4 h-4" />
            <span className="text-sm font-medium">Share</span>
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

      <div className="mb-6">
        <div className="flex items-center space-x-2 mb-4">
          <Sparkles className="w-5 h-5 text-blue-600" />
          <h3 className="text-xl font-bold text-gray-900">Synthesized Answer</h3>
        </div>
        <div className="bg-white rounded-xl p-8 shadow-sm border border-gray-200">
          <div className="prose prose-lg max-w-none">
            <p className="text-gray-800 leading-relaxed whitespace-pre-wrap text-lg">{answer}</p>
          </div>
        </div>
      </div>

      {/* Success Footer */}
      <div className="pt-6 border-t border-green-200">
        <div className="flex items-center justify-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Research completed successfully</span>
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

export default AnswerDisplay;