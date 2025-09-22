import React, { useState, useRef } from 'react';
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
import { Upload, File, X, CheckCircle2 } from 'lucide-react';

interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
}

interface DocumentUploadProps {
  onDocumentsChange: (documents: UploadedDocument[]) => void;
  onDocumentResearch?: (file: File) => void;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({ onDocumentsChange, onDocumentResearch }) => {
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [originalFiles, setOriginalFiles] = useState<Map<string, File>>(new Map());
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (files: FileList | null) => {
    if (!files) return;

    const validFiles = Array.from(files).filter(file => {
      const validTypes = ['.txt', '.pdf', '.docx'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
      return validTypes.includes(fileExtension);
    });

    if (validFiles.length === 0) {
      alert('Please select valid files (.txt, .pdf, .docx)');
      return;
    }

    try {
      // Upload files to backend
      const formData = new FormData();
      validFiles.forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Update documents with backend response
      const newDocuments: UploadedDocument[] = result.documents.map((doc: any) => ({
        id: doc.id,
        name: doc.name,
        size: doc.size,
        type: doc.type
      }));

      // Store original files for research
      const newOriginalFiles = new Map(originalFiles);
      validFiles.forEach((file, index) => {
        if (index < newDocuments.length) {
          newOriginalFiles.set(newDocuments[index].id, file);
        }
      });
      setOriginalFiles(newOriginalFiles);

      const updatedDocuments = [...documents, ...newDocuments];
      setDocuments(updatedDocuments);
      onDocumentsChange(updatedDocuments);
    } catch (error) {
      console.error('Error uploading documents:', error);
      // Fallback: add documents locally without backend processing
      const newDocuments: UploadedDocument[] = validFiles.map(file => ({
        id: Math.random().toString(36).substr(2, 9),
        name: file.name,
        size: file.size,
        type: file.type
      }));

      // Store original files for research (fallback case)
      const newOriginalFiles = new Map(originalFiles);
      validFiles.forEach((file, index) => {
        newOriginalFiles.set(newDocuments[index].id, file);
      });
      setOriginalFiles(newOriginalFiles);

      const updatedDocuments = [...documents, ...newDocuments];
      setDocuments(updatedDocuments);
      onDocumentsChange(updatedDocuments);
      
      alert('Documents uploaded locally. For full processing, ensure the backend server is running.');
    }
  };

  const removeDocument = (id: string) => {
    const updatedDocuments = documents.filter(doc => doc.id !== id);
    const newOriginalFiles = new Map(originalFiles);
    newOriginalFiles.delete(id);
    setOriginalFiles(newOriginalFiles);
    setDocuments(updatedDocuments);
    onDocumentsChange(updatedDocuments);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  return (
    <div className="w-full max-w-4xl mx-auto mb-8">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Upload className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Upload Documents</h3>
          <span className="text-sm text-gray-500">(Optional)</span>
        </div>
        
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragging 
              ? 'border-blue-400 bg-blue-50' 
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.pdf,.docx"
            onChange={(e) => handleFileSelect(e.target.files)}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <div className="space-y-4">
            <div className="w-12 h-12 mx-auto bg-blue-100 rounded-lg flex items-center justify-center">
              <Upload className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900 mb-2">
                Drop files here or click to browse
              </p>
              <p className="text-sm text-gray-500">
                Supports .txt, .pdf, .docx files up to 10MB each
              </p>
            </div>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>Choose Files</span>
            </button>
          </div>
        </div>

        {documents.length > 0 && (
          <div className="mt-6">
            <h4 className="text-sm font-medium text-gray-900 mb-3">
              Uploaded Documents ({documents.length})
            </h4>
            <div className="space-y-2">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                      <File className="w-4 h-4 text-green-600" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900">{doc.name}</p>
                      <p className="text-xs text-gray-500">{formatFileSize(doc.size)}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                    {onDocumentResearch && originalFiles.has(doc.id) && (
                      <button
                        onClick={() => onDocumentResearch(originalFiles.get(doc.id)!)}
                        className="px-3 py-1 bg-blue-600 text-white text-xs rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        Research
                      </button>
                    )}
                    <button
                      onClick={() => removeDocument(doc.id)}
                      className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentUpload;