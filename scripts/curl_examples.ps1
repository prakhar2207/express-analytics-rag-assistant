# PowerShell script to test all FastAPI endpoints against http://127.0.0.1:8000

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Testing FastAPI Endpoints (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Root
Write-Host "`n1. Testing GET / (Health)" -ForegroundColor Yellow
$rootResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/" -Method Get
$rootResp | ConvertTo-Json -Depth 3

# 2. Documents
Write-Host "`n2. Testing GET /documents (List Corpus)" -ForegroundColor Yellow
$docsResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents" -Method Get
Write-Host "Total Documents: $($docsResp.total_documents), Total Chunks: $($docsResp.total_chunks)"
$docsResp.documents | ForEach-Object { Write-Host " - $($_.source) ($($_.chunks_count) chunks)" }

# 3. Query
Write-Host "`n3. Testing POST /query (Q&A with Citations)" -ForegroundColor Yellow
$queryBody = @{
    question = "How do I configure CORS middleware in FastAPI with specific origins?"
    session_id = "powershell_demo"
} | ConvertTo-Json

$queryResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" -Method Post -Body $queryBody -ContentType "application/json"
Write-Host "Intent: $($queryResp.query_type) | Execution: $($queryResp.execution_time_sec)s | Grounded: $($queryResp.is_grounded)"
Write-Host "Answer: $($queryResp.answer)"
Write-Host "Citations:"
$queryResp.sources | ForEach-Object { Write-Host " - [$($_.source) | $($_.section)]" }

# 4. Feedback
Write-Host "`n4. Testing POST /feedback (Record Rating)" -ForegroundColor Yellow
$feedbackBody = @{
    query = "How do I configure CORS middleware?"
    answer = "Use CORSMiddleware..."
    rating = "up"
    comment = "Great response!"
} | ConvertTo-Json

$feedbackResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/feedback" -Method Post -Body $feedbackBody -ContentType "application/json"
$feedbackResp | ConvertTo-Json

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " All Endpoints Tested Successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
