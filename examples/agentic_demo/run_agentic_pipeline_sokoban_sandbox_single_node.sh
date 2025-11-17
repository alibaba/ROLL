#!/bin/bash
set +x


echo "Stopping existing services..."
pkill -9 -f "admin --env local" 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 2

echo "Starting ROCK service..."
cd /workspace/ROCK && nohup admin --env local > /tmp/rock_service.log 2>&1 &

echo "Waiting for ROCK service to be ready..."
sleep 5

cd $PWD

CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$PWD:$PYTHONPATH"
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agent_val_sokoban_sandbox

