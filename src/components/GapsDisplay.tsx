import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

interface GapsDisplayProps {
  gaps: string[];
}

const GapsDisplay: React.FC<GapsDisplayProps> = ({ gaps }) => {
  if (gaps.length === 0) return null;

  return (
    <div className="bg-amber-50 border-2 border-amber-200 rounded-xl p-6">
      <div className="flex items-center space-x-2 mb-4">
        <AlertTriangle className="w-6 h-6 text-amber-600" />
        <h3 className="text-lg font-semibold text-amber-900">Knowledge Gaps Identified</h3>
      </div>
      
      <div className="space-y-3">
        {gaps.map((gap, index) => (
          <div key={index} className="flex items-start space-x-3">
            <Info className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
            <p className="text-amber-800 text-sm">{gap}</p>
          </div>
        ))}
      </div>
      
      <div className="mt-4 p-3 bg-amber-100 rounded-lg">
        <p className="text-xs text-amber-700">
          <strong>Note:</strong> These gaps indicate areas where additional research or data sources 
          might be needed for a more comprehensive analysis.
        </p>
      </div>
    </div>
  );
};

export default GapsDisplay;