import React from 'react';
import { Brain, Search, FileText, Zap } from 'lucide-react';

interface LoadingStateProps {
  currentStep: number;
  query: string;
}

const LoadingState: React.FC<LoadingStateProps> = ({ currentStep, query }) => {
  const steps = [
    {
      id: 1,
      title: 'Query Decomposition',
      subtitle: 'Breaking down into sub-questions',
      description: 'Analyzing your query and decomposing it into focused sub-questions for comprehensive research',
      icon: Brain,
      color: 'blue'
    },
    {
      id: 2,
      title: 'Document Retrieval',
      subtitle: 'Retrieving relevant documents',
      description: 'Searching through knowledge bases and uploaded documents to find the most relevant information',
      icon: Search,
      color: 'purple'
    },
    {
      id: 3,
      title: 'Synthesis & Analysis',
      subtitle: 'Combining findings into final report',
      description: 'Processing retrieved information, identifying gaps, and synthesizing insights into a comprehensive answer',
      icon: FileText,
      color: 'green'
    }
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Deep Research Agent Working</h2>
        <p className="text-lg text-gray-600 mb-6">
          Conducting multi-agent research with full chain-of-thought transparency
        </p>
        <div className="bg-blue-50 rounded-lg p-4 max-w-2xl mx-auto">
          <p className="text-blue-800 font-medium">"{query}"</p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="relative mb-12">
        <div className="absolute top-1/2 transform -translate-y-1/2 w-full h-2 bg-gray-200 rounded-full">
          <div 
            className="h-full bg-blue-600 rounded-full transition-all duration-700 ease-out"
            style={{ width: `${(currentStep / steps.length) * 100}%` }}
          ></div>
        </div>
        <div className="relative flex justify-between">
          {steps.map((step) => (
            <div
              key={step.id}
              className={`flex items-center justify-center w-12 h-12 rounded-full border-4 border-white shadow-lg transition-all duration-300 ${
                currentStep >= step.id 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-400'
              }`}
            >
              <step.icon className="w-5 h-5" />
            </div>
          ))}
        </div>
      </div>

      {/* Step Cards */}
      <div className="grid gap-6 md:grid-cols-3">
        {steps.map((step) => {
          const isActive = currentStep === step.id;
          const isCompleted = currentStep > step.id;
          const IconComponent = step.icon;

          return (
            <div
              key={step.id}
              className={`relative p-6 rounded-xl border-2 transition-all duration-300 ${
                isActive
                  ? 'bg-blue-50 border-blue-200 shadow-lg transform scale-105'
                  : isCompleted
                  ? 'bg-green-50 border-green-200'
                  : 'bg-gray-50 border-gray-200'
              }`}
            >
              <div className={`w-12 h-12 rounded-xl mb-4 flex items-center justify-center ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : isCompleted
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-300 text-gray-500'
              }`}>
                <IconComponent className="w-6 h-6" />
              </div>
              
              <h3 className={`font-semibold text-lg mb-2 ${
                isActive ? 'text-blue-900' : isCompleted ? 'text-green-900' : 'text-gray-700'
              }`}>
                {step.title}
              </h3>
              
              <p className={`text-sm font-medium mb-2 ${
                isActive ? 'text-blue-700' : isCompleted ? 'text-green-700' : 'text-gray-500'
              }`}>
                {step.subtitle}
              </p>
              
              <p className={`text-sm ${
                isActive ? 'text-blue-600' : isCompleted ? 'text-green-600' : 'text-gray-500'
              }`}>
                {step.description}
              </p>

              {isActive && (
                <div className="absolute -top-2 -right-2">
                  <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                    <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                  </div>
                </div>
              )}

              {isCompleted && (
                <div className="absolute -top-2 -right-2">
                  <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Processing Message */}
      <div className="text-center mt-12">
        <div className="inline-flex items-center space-x-2 bg-white rounded-full px-6 py-3 shadow-sm border border-gray-200">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-75"></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-150"></div>
          </div>
          <span className="text-gray-600 ml-3 font-medium">
            {currentStep <= steps.length 
              ? `${steps[currentStep - 1]?.title} in progress...` 
              : 'Finalizing research...'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default LoadingState;