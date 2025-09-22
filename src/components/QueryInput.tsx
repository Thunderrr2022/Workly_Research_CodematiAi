import React, { useState } from 'react';
import { Search, Sparkles, FileText } from 'lucide-react';

interface QueryInputProps {
  onQuery: (query: string) => void;
  hasDocuments?: boolean;
}

const QueryInput: React.FC<QueryInputProps> = ({ onQuery, hasDocuments = false }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onQuery(input.trim());
    }
  };

  const exampleQueries = [
    "How is artificial intelligence transforming healthcare?",
    "What are the environmental impacts of renewable energy adoption?",
    "Analyze the economic effects of remote work on urban planning"
  ];

  return (
    <div className="max-w-4xl mx-auto">
      {hasDocuments && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <FileText className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-800 font-medium">
              Documents uploaded - your research will include these files
            </span>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={hasDocuments ? "Ask a question about your documents..." : "Enter your research question..."}
            className="w-full pl-12 pr-32 py-4 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-colors"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            <Sparkles className="w-4 h-4" />
            <span>Research</span>
          </button>
        </div>
      </form>

      <div className="mt-6">
        <p className="text-sm text-gray-500 mb-3">Try these examples:</p>
        <div className="flex flex-wrap gap-3">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => setInput(example)}
              className="text-sm text-blue-600 bg-blue-50 hover:bg-blue-100 px-3 py-2 rounded-lg transition-colors"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QueryInput;