#!/bin/bash
# save as smoke_test.sh → chmod +x smoke_test.sh → ./smoke_test.sh

BASE="http://localhost:8000"
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== DocuMind Smoke Test ==="

# 1. Health
echo -n "GET  /health              → "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE/health)
[ "$STATUS" = "200" ] && echo -e "${GREEN}$STATUS OK${NC}" || echo -e "${RED}$STATUS FAIL${NC}"

# 2. List documents (empty)
echo -n "GET  /documents           → "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE/documents)
[ "$STATUS" = "200" ] && echo -e "${GREEN}$STATUS OK${NC}" || echo -e "${RED}$STATUS FAIL${NC}"

# 3. Ingest a test file
echo -n "POST /ingest              → "
echo "This is a test document. It contains information about payment terms: net 30 days." > /tmp/test_doc.txt
INGEST=$(curl -s -X POST $BASE/ingest -F "file=@/tmp/test_doc.txt")
STATUS=$(echo $INGEST | jq -r '.status')
DOC_ID=$(echo $INGEST | jq -r '.doc_id')
[ "$STATUS" = "ready" ] && echo -e "${GREEN}201 OK — doc_id=$DOC_ID${NC}" || echo -e "${RED}FAIL — $INGEST${NC}"

# 4. Verify document appears
echo -n "GET  /documents/{doc_id}  → "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE/documents/$DOC_ID)
[ "$STATUS" = "200" ] && echo -e "${GREEN}$STATUS OK${NC}" || echo -e "${RED}$STATUS FAIL${NC}"

# 5. Query
echo -n "POST /query               → "
QUERY=$(curl -s -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}')
HAS_CTX=$(echo $QUERY | jq -r '.has_relevant_context')
[ "$HAS_CTX" = "true" ] && echo -e "${GREEN}200 OK — has_context=true${NC}" || echo -e "${RED}FAIL — $QUERY${NC}"

# 6. Delete
echo -n "DELETE /documents/{doc_id}→ "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE $BASE/documents/$DOC_ID)
[ "$STATUS" = "200" ] && echo -e "${GREEN}$STATUS OK${NC}" || echo -e "${RED}$STATUS FAIL${NC}"

# Cleanup
rm /tmp/test_doc.txt

echo "==========================="
echo "Smoke test complete"