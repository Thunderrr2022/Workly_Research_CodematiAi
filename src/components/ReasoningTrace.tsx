import React from 'react';
import { CheckCircle2, Brain, Search, FileText, Zap, AlertTriangle } from 'lucide-react';

interface ReasoningTraceProps {
  steps: string[];
}

const ReasoningTrace: React.FC<ReasoningTraceProps> = ({ steps }) => {
  const getStepIcon = (step: string, index: number) => {
    if (step.toLowerCase().includes('sub-question') || step.toLowerCase().includes('decompos')) {
      return Brain;
    }
    if (step.toLowerCase().includes('retriev') || step.toLowerCase().includes('document')) {
      return Search;
    }
    if (step.toLowerCase().includes('summar') || step.toLowerCase().includes('synthes')) {
      return FileText;
    }
    if (step.toLowerCase().includes('gap') || step.toLowerCase().includes('missing')) {
      return AlertTriangle;
    }
    const icons = [Brain, Search, FileText, Zap];
    const IconComponent = icons[index] || CheckCircle2;
    return IconComponent;
  };

  const getStepTitle = (step: string) => {
    if (step.toLowerCase().includes('sub-question') || step.toLowerCase().includes('decompos')) {
      return 'Query Decomposition';
    }
    if (step.toLowerCase().includes('retriev') || step.toLowerCase().includes('document')) {
      return 'Document Retrieval';
    }
    if (step.toLowerCase().includes('summar') || step.toLowerCase().includes('synthes')) {
      return 'Information Synthesis';
    }
    if (step.toLowerCase().includes('gap') || step.toLowerCase().includes('missing')) {
      return 'Knowledge Gap Analysis';
    }
    return 'Processing Step';
  };

  const getStepColor = (step: string) => {
    if (step.toLowerCase().includes('gap') || step.toLowerCase().includes('missing')) {
      return 'amber';
    }
    return 'blue';
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-6">
        <Brain className="w-6 h-6 text-blue-600" />
        <h2 className="text-xl font-semibold text-gray-900">Research Process</h2>
      </div>

      {/* Progress Bar */}
      <div className="relative mb-8">
        <div className="absolute top-1/2 transform -translate-y-1/2 w-full h-2 bg-gray-200 rounded-full">
          <div 
            className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-1000 ease-out"
            style={{ width: '100%' }}
          ></div>
        </div>
        <div className="relative flex justify-between">
          {steps.map((_, index) => (
            <div
              key={index}
              className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 border-4 border-white rounded-full shadow-lg transform hover:scale-110 transition-all duration-300"
              style={{
                animationDelay: `${index * 150}ms`,
                animation: 'pulse 0.6s ease-out forwards'
              }}
            >
              <span className="text-sm font-bold text-white">{index + 1}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-6">
        {steps.map((step, index) => {
          const IconComponent = getStepIcon(step, index);
          const isActive = true; // All steps are completed
          const isCompleted = true;
          const stepColor = getStepColor(step);
          const colorClasses = stepColor === 'amber' 
            ? 'bg-amber-50 border-amber-200 text-amber-900'
            : 'bg-blue-50 border-blue-200 text-blue-900';
          const iconColorClasses = stepColor === 'amber'
            ? 'bg-amber-600 text-white'
            : 'bg-blue-600 text-white';

          return (
            <div
              key={index}
              className={`relative flex items-start space-x-4 p-6 rounded-xl transition-all duration-500 transform hover:scale-[1.02] ${
                isActive ? colorClasses : 'bg-gray-50 border-2 border-gray-200'
              }`}
              style={{
                animationDelay: `${index * 200}ms`,
                animation: 'slideInUp 0.6s ease-out forwards'
              }}
            >
              {/* Step Number Badge */}
              <div className="absolute -top-3 -left-3 w-8 h-8 bg-white border-2 border-blue-600 rounded-full flex items-center justify-center shadow-sm">
                <span className="text-sm font-bold text-blue-600">{index + 1}</span>
              </div>
              
              <div className={`flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center shadow-sm ${
                isActive ? iconColorClasses : 'bg-gray-400 text-white'
              }`}>
                <IconComponent className="w-6 h-6" />
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2 mb-2">
                  <h3 className={`text-lg font-bold ${
                    isActive ? (stepColor === 'amber' ? 'text-amber-900' : 'text-blue-900') : 'text-gray-700'
                  }`}>
                    {getStepTitle(step)}
                  </h3>
                  {isCompleted && (
                    <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                      <CheckCircle2 className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
                <p className={`leading-relaxed ${
                  isActive ? (stepColor === 'amber' ? 'text-amber-800' : 'text-blue-800') : 'text-gray-600'
                }`}>
                  {step}
                </p>
              </div>
            </div>
          );
        })}
      </div>
      
      <style>{`
        @keyframes slideInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes pulse {
          0% {
            opacity: 0;
            transform: scale(0.8);
          }
          50% {
            opacity: 1;
            transform: scale(1.1);
          }
          100% {
            opacity: 1;
            transform: scale(1);
          }
        }
      `}</style>
    </div>
  );
};

export default ReasoningTrace;