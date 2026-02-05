#!/usr/bin/env bash
set -euo pipefail

# Usage: ./client.sh <base_url> <file_path> <schema_path>
BASE_URL="${1:?Usage: client.sh <base_url> <file_path> <schema_path>}"
FILE_PATH="$(cd "$(dirname "${2:?}")" && pwd)/$(basename "$2")"
SCHEMA_PATH="$(cd "$(dirname "${3:?}")" && pwd)/$(basename "$3")"
FILE_NAME="$(basename "$FILE_PATH")"

echo "Uploading: $FILE_NAME"
echo "Schema:    $SCHEMA_PATH"
echo "Endpoint:  $BASE_URL"
echo ""

# POST file + schema → get task_id
RESPONSE=$(docker run --rm --network host \
  -v "$FILE_PATH:/upload/$FILE_NAME" \
  -v "$SCHEMA_PATH:/upload/schema.json" \
  curlimages/curl -s -X POST "$BASE_URL/tasks" \
    -F "file=@/upload/$FILE_NAME" \
    -F "schema=</upload/schema.json")

TASK_ID=$(echo "$RESPONSE" | docker run --rm -i python:3.11-alpine \
  python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

echo "Task created: $TASK_ID"
echo "Polling every 10 seconds..."
echo ""

while true; do
  sleep 10

  STATUS_JSON=$(docker run --rm --network host \
    curlimages/curl -s "$BASE_URL/tasks/$TASK_ID")

  STATUS=$(echo "$STATUS_JSON" | docker run --rm -i python:3.11-alpine \
    python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")

  case "$STATUS" in
    succeeded)
      echo "=== EXTRACTION SUCCEEDED ==="
      echo ""
      docker run --rm --network host \
        curlimages/curl -s "$BASE_URL/tasks/$TASK_ID/result" | \
        docker run --rm -i python:3.11-alpine \
        python3 -c "import sys,json; json.dump(json.load(sys.stdin),sys.stdout,indent=2)"
      echo ""
      exit 0
      ;;
    failed)
      echo "=== EXTRACTION FAILED ===" >&2
      echo "$STATUS_JSON" | docker run --rm -i python:3.11-alpine \
        python3 -c "import sys,json; json.dump(json.load(sys.stdin),sys.stdout,indent=2)" >&2
      echo "" >&2
      exit 1
      ;;
    *)
      echo "  [$(date +%H:%M:%S)] status: $STATUS"
      ;;
  esac
done
