#!/usr/bin/env bash
# Bash script to test all FastAPI endpoints against http://127.0.0.1:8000

set -e

echo "============================================================"
echo " Testing FastAPI Endpoints (Bash / cURL)"
echo "============================================================"

echo -e "\n1. Testing GET / (Health)"
curl -s -X GET "http://127.0.0.1:8000/" | python -m json.tool

echo -e "\n2. Testing GET /documents (List Corpus)"
curl -s -X GET "http://127.0.0.1:8000/documents" | python -m json.tool

echo -e "\n3. Testing POST /query (Q&A with Citations)"
curl -s -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I configure CORS middleware in FastAPI with specific origins?", "session_id": "curl_demo"}' | python -m json.tool

echo -e "\n4. Testing POST /feedback (Record Rating)"
curl -s -X POST "http://127.0.0.1:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{"query": "CORS query", "answer": "Use CORSMiddleware", "rating": "up", "comment": "Verified via curl"}' | python -m json.tool

echo -e "\n============================================================"
echo " All Endpoints Tested Successfully!"
echo "============================================================"
