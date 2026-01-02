#!/bin/bash
# Start script for Polymarket Whale Tracker

set -e

echo "==================================="
echo "Polymarket Whale Tracker"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "whale_tracker.py" ]; then
    echo "Error: Run this script from the predmkt directory"
    exit 1
fi

# Function to kill background processes on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
echo ""
echo "[1/2] Starting backend server..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Backend failed to start"
    exit 1
fi

# Start frontend
echo ""
echo "[2/2] Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "==================================="
echo "Services started:"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo "==================================="
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait
