#!/bin/sh

# Check if PORT is set, otherwise default to 8000
PORT=${PORT:-8000}

# Start gunicorn with the specified port
exec gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 8 --timeout 0 main:app
