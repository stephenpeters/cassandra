#!/bin/bash
# Start Cassandra backend with VPN connection
# Requires ExpressVPN CLI to be installed

set -e

# Check if ExpressVPN is connected
check_vpn() {
    if command -v expressvpn &> /dev/null; then
        STATUS=$(expressvpn status 2>/dev/null || echo "Not connected")
        if [[ "$STATUS" == *"Connected"* ]]; then
            echo "[VPN] Already connected"
            return 0
        fi
    fi
    return 1
}

# Connect to VPN
connect_vpn() {
    if command -v expressvpn &> /dev/null; then
        echo "[VPN] Connecting to Ireland..."
        expressvpn connect "Ireland" || expressvpn connect "UK - London"
        sleep 3

        if check_vpn; then
            echo "[VPN] Connected successfully"
            return 0
        fi
    fi

    echo "[VPN] WARNING: ExpressVPN not available or failed to connect"
    echo "[VPN] Polymarket API calls may be blocked without VPN"
    return 1
}

# Main
cd "$(dirname "$0")"
source venv/bin/activate

# Try to connect VPN (non-fatal if it fails)
check_vpn || connect_vpn || echo "[VPN] Continuing without VPN..."

# Start server
echo "[Server] Starting Cassandra backend..."
exec python -m uvicorn server:app --host 127.0.0.1 --port 8000
