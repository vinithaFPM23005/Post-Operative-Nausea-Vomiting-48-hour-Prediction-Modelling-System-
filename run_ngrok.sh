#!/usr/bin/env bash
# Starts Streamlit and exposes it through an ngrok HTTPS URL for opening in Chrome on a phone.
set -e
cd "$(dirname "$0")"
command -v ngrok >/dev/null || { echo "Install ngrok and run: ngrok config add-authtoken <TOKEN>"; exit 1; }

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
APP_PID=$!
trap 'kill $APP_PID' EXIT
sleep 5
ngrok http 8501
