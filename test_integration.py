#!/usr/bin/env python3
"""
Deep Researcher Agent - Integration Test Script
Tests the full end-to-end functionality of the system.
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
TEST_QUERY = "How is artificial intelligence transforming healthcare?"
TEST_DOCUMENT_CONTENT = """
Artificial Intelligence in Healthcare: A Comprehensive Analysis

Artificial intelligence (AI) is revolutionizing healthcare through various applications:

1. Diagnostic Imaging: AI systems can analyze medical images with 94% accuracy, often surpassing human radiologists in detecting early-stage diseases.

2. Drug Discovery: Machine learning algorithms have reduced drug discovery timelines from 10+ years to 3-5 years, significantly lowering development costs.

3. Personalized Medicine: AI analyzes patient data to create individualized treatment plans, improving outcomes by 23% on average.

4. Predictive Analytics: Healthcare systems use AI to predict patient deterioration, reducing emergency interventions by 35%.

The integration of AI in healthcare represents a paradigm shift toward more precise, efficient, and accessible medical care.
"""

def create_test_document():
    """Create a test document for upload."""
    test_doc_path = Path("test_document.txt")
    with open(test_doc_path, "w", encoding="utf-8") as f:
        f.write(TEST_DOCUMENT_CONTENT)
    return test_doc_path

def test_backend_health():
    """Test if backend is running and healthy."""
    print("🔍 Testing backend health...")
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend is not running: {e}")
        return False

def test_document_upload():
    """Test document upload functionality."""
    print("📄 Testing document upload...")
    
    # Create test document
    test_doc = create_test_document()
    
    try:
        with open(test_doc, "rb") as f:
            files = {"files": ("test_document.txt", f, "text/plain")}
            response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document uploaded successfully: {result['message']}")
            print(f"   Documents: {len(result['documents'])}")
            return result['documents'][0]['id'] if result['documents'] else None
        else:
            print(f"❌ Document upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Document upload error: {e}")
        return None
    finally:
        # Clean up test document
        if test_doc.exists():
            test_doc.unlink()

def test_query_processing(doc_id=None):
    """Test query processing functionality."""
    print("🔍 Testing query processing...")
    
    try:
        payload = {
            "query": TEST_QUERY,
            "documents": [doc_id] if doc_id else []
        }
        
        response = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query processed successfully")
            print(f"   Answer length: {len(result['answer'])} characters")
            print(f"   Reasoning steps: {len(result['reasoning_trace'])}")
            print(f"   Knowledge gaps: {len(result['gaps'])}")
            
            # Print first part of answer
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"   Answer preview: {answer_preview}")
            
            return result
        else:
            print(f"❌ Query processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Query processing error: {e}")
        return None

def test_report_export(query_result):
    """Test report export functionality."""
    if not query_result:
        print("⏭️  Skipping report export test (no query result)")
        return False
    
    print("📊 Testing report export...")
    
    try:
        # Test PDF export
        payload = {
            "query": TEST_QUERY,
            "answer": query_result['answer'],
            "reasoning_trace": query_result['reasoning_trace'],
            "gaps": query_result['gaps'],
            "documents": ["test_document.txt"],
            "format": "pdf"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/export",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ PDF export successful")
            print(f"   File size: {len(response.content)} bytes")
        else:
            print(f"❌ PDF export failed: {response.status_code}")
            return False
        
        # Test Markdown export
        payload["format"] = "markdown"
        response = requests.post(
            f"{BACKEND_URL}/export",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Markdown export successful")
            print(f"   File size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Markdown export failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Report export error: {e}")
        return False

def test_document_management():
    """Test document management functionality."""
    print("📚 Testing document management...")
    
    try:
        # List documents
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document listing successful: {len(result['documents'])} documents")
            return result['documents']
        else:
            print(f"❌ Document listing failed: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Document management error: {e}")
        return []

def main():
    """Run the complete integration test suite."""
    print("🚀 Deep Researcher Agent - Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Backend Health
    if not test_backend_health():
        print("\n❌ Backend is not running. Please start it with:")
        print("   cd backend && python main.py")
        return False
    
    print()
    
    # Test 2: Document Upload
    doc_id = test_document_upload()
    print()
    
    # Test 3: Query Processing
    query_result = test_query_processing(doc_id)
    print()
    
    # Test 4: Report Export
    export_success = test_report_export(query_result)
    print()
    
    # Test 5: Document Management
    documents = test_document_management()
    print()
    
    # Summary
    print("=" * 50)
    print("📋 Test Summary:")
    print(f"   Backend Health: {'✅' if True else '❌'}")
    print(f"   Document Upload: {'✅' if doc_id else '❌'}")
    print(f"   Query Processing: {'✅' if query_result else '❌'}")
    print(f"   Report Export: {'✅' if export_success else '❌'}")
    print(f"   Document Management: {'✅' if documents is not None else '❌'}")
    
    all_tests_passed = all([True, doc_id, query_result, export_success, documents is not None])
    
    if all_tests_passed:
        print("\n🎉 All tests passed! The Deep Researcher Agent is fully functional.")
        print("\n🌐 Frontend Integration:")
        print("   1. Start the frontend: npm run dev")
        print("   2. Open http://localhost:5173")
        print("   3. Upload documents and try queries!")
    else:
        print("\n⚠️  Some tests failed. Check the backend logs for details.")
        print("   Backend logs: Check console output where backend is running")
    
    return all_tests_passed

if __name__ == "__main__":
    main()


