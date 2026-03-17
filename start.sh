#!/bin/bash

PORT=${1:-8000}

echo "Checking for existing process on port $PORT..."
PID=$(lsof -t -i:$PORT)

if [ ! -z "$PID" ]; then
    echo "Killing process $PID running on port $PORT..."
    kill -9 $PID
    sleep 1
fi

echo "Starting Research Assistant server on port $PORT..."
source .venv/bin/activate
uvicorn src.research_assistant.api.server:app --reload --host 0.0.0.0 --port $PORT
