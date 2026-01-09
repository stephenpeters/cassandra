# Production Deployment Guide

## Vultr VPS Setup (Recommended: High Frequency in NYC or Amsterdam)

### 1. Server Requirements
- **Minimum**: 2 vCPU, 4GB RAM, 50GB SSD
- **Recommended**: 4 vCPU, 8GB RAM, 100GB SSD
- **Location**: NYC (closest to Polymarket) or Amsterdam
- **OS**: Ubuntu 22.04 LTS

### 2. Initial Server Hardening

```bash
# Update system
apt update && apt upgrade -y

# Create non-root user
adduser predmkt
usermod -aG sudo predmkt

# SSH hardening
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Setup firewall (only SSH + internal ports)
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 8000/tcp  # API (will be behind nginx)
ufw enable

# Install fail2ban
apt install fail2ban -y
systemctl enable fail2ban
```

### 3. Application Setup

```bash
# Install dependencies
apt install python3.11 python3.11-venv python3-pip nginx certbot python3-certbot-nginx -y

# Clone repo
cd /opt
git clone <your-repo> predmkt
chown -R predmkt:predmkt predmkt

# Setup Python environment
cd /opt/predmkt/backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Systemd Service (Auto-restart)

Create `/etc/systemd/system/predmkt.service`:

```ini
[Unit]
Description=Predmkt Paper Trading Server
After=network.target

[Service]
Type=simple
User=predmkt
Group=predmkt
WorkingDirectory=/opt/predmkt/backend
Environment="PATH=/opt/predmkt/backend/venv/bin"
EnvironmentFile=/opt/predmkt/.env
ExecStart=/opt/predmkt/backend/venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/predmkt/backend

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable predmkt
systemctl start predmkt
```

### 5. Nginx Reverse Proxy with Auth

Create `/etc/nginx/sites-available/predmkt`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Basic auth for API access
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket timeout
        proxy_read_timeout 86400;
    }
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

```bash
# Create htpasswd file
apt install apache2-utils -y
htpasswd -c /etc/nginx/.htpasswd admin

# Enable site
ln -s /etc/nginx/sites-available/predmkt /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# Get SSL cert
certbot --nginx -d your-domain.com
```

### 6. Secrets Management

Create `/opt/predmkt/.env`:

```bash
# API Keys (if needed in future for live trading)
# POLYMARKET_API_KEY=xxx
# POLYMARKET_API_SECRET=xxx

# Server settings
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Logging
LOG_LEVEL=INFO
```

```bash
chmod 600 /opt/predmkt/.env
chown predmkt:predmkt /opt/predmkt/.env
```

### 7. Monitoring & Alerts

```bash
# Install monitoring
apt install prometheus-node-exporter -y

# Setup log rotation
cat > /etc/logrotate.d/predmkt << EOF
/var/log/predmkt/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

### 8. Health Check Script

Create `/opt/predmkt/healthcheck.sh`:

```bash
#!/bin/bash
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
if [ "$RESPONSE" != "200" ]; then
    echo "Health check failed, restarting..."
    systemctl restart predmkt
fi
```

Add to crontab:
```bash
*/5 * * * * /opt/predmkt/healthcheck.sh >> /var/log/predmkt/healthcheck.log 2>&1
```

### 9. Backup Strategy

```bash
# Daily backup of trading state
0 0 * * * tar -czf /opt/backups/predmkt-$(date +\%Y\%m\%d).tar.gz /opt/predmkt/backend/paper_trading_state.json
```

## Live Trading Wallet Setup

### Overview: Two Wallet Setup Options

Polymarket supports two wallet configurations for API trading:

| Method | Signature Type | Best For |
|--------|---------------|----------|
| **Proxy Wallet** | 2 | **Recommended** - use Polymarket.com deposit flow |
| **EOA Direct** | 0 | Simple setup, funds directly in wallet |

### Option A: Proxy Wallet Setup (Recommended)

This uses Polymarket's proxy wallet system, created when you deposit via their website.

```bash
# 1. Create a fresh MetaMask wallet (don't use your main wallet!)
#    - Install MetaMask from https://metamask.io
#    - Create new wallet, save seed phrase securely
#    - Add Polygon network (Chain ID: 137, RPC: https://polygon-rpc.com)

# 2. Go to https://polymarket.com and connect your MetaMask

# 3. Click "Deposit" - this creates your proxy wallet
#    Note the deposit address shown - this is your PROXY WALLET address

# 4. Fund via card, crypto, or bridge

# Example addresses after setup:
# - MetaMask (signer): 0x55e7A3896E4f790a6F111b71B7F99d4190E11298
# - Proxy (maker):     0x21f6163c35B3B2523a4db1cf61B33E55b8e071b1
```

**Configure .env for Proxy Wallet:**
```bash
POLYMARKET_PRIVATE_KEY=your_metamask_private_key_no_0x_prefix
POLYMARKET_SIGNATURE_TYPE=2
POLYMARKET_FUNDER=0xYourProxyWalletAddress
```

### Option B: EOA Direct Setup

Fund your MetaMask wallet directly with USDC.e (not via Polymarket website).

```bash
# 1. Create a fresh MetaMask wallet
# 2. Add Polygon network (Chain ID: 137)
# 3. Transfer POL (for gas) and USDC.e directly to this wallet

# Generate wallet via Python (alternative):
python3 -c "from eth_account import Account; a=Account.create(); print(f'Address: {a.address}\\nPrivate Key: {a.key.hex()[2:]}')"
```

**Configure .env for EOA Direct:**
```bash
POLYMARKET_PRIVATE_KEY=your_private_key_no_0x_prefix
POLYMARKET_SIGNATURE_TYPE=0
# No POLYMARKET_FUNDER needed - the signer address is used as maker
```

### 2. Fund Your Wallet with POL (for gas)

You need POL (formerly MATIC) for transaction gas fees on Polygon.

**Getting POL:**
- Buy POL on any exchange (Coinbase, Binance, Kraken)
- Withdraw to your wallet address on **Polygon network** (not Ethereum!)
- Minimum recommended: **10 POL** (~$5 at current prices)
- Gas per trade is ~0.001 POL (~$0.01)

### 3. Fund Your Wallet with USDC.e (for trading)

**CRITICAL: Polymarket uses USDC.e, NOT native USDC**

| Token | Contract Address | Works on Polymarket? |
|-------|-----------------|---------------------|
| USDC.e (Bridged) | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | YES |
| USDC (Native) | `0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359` | NO |

**Getting USDC.e:**
1. **From an exchange**: Withdraw USDC to Polygon (most exchanges send USDC.e)
2. **From Polymarket**: Withdraw from your Polymarket account to your trading wallet
3. **Swap**: Use [QuickSwap](https://quickswap.exchange) or [Uniswap](https://app.uniswap.org) to swap native USDC → USDC.e

**Why USDC.e?**
- USDC.e is the bridged version from Ethereum (the original on Polygon)
- Native USDC arrived later and Polymarket's contracts use the original USDC.e
- If you have native USDC, swap it on a DEX before trading

### 4. Configure the .env File

```bash
# In backend/.env
POLYMARKET_PRIVATE_KEY=your_64_char_hex_private_key_no_0x_prefix
POLYMARKET_SIGNATURE_TYPE=0
POLYMARKET_FUNDER=0xYourWalletAddressHere
```

**Signature Types:**
- `0` = EOA (Externally Owned Account) - **Use this for MetaMask wallets**
- `1` = Poly Proxy
- `2` = Gnosis Safe

### 5. Set Token Allowances (First-Time Setup)

Before your first trade, you must approve Polymarket contracts to spend your tokens:

1. Switch to **Live Mode** in the UI
2. Enter your API key
3. Click **"Set Allowances"**

This sends blockchain transactions to approve:
- USDC.e spending on the exchange contract
- Conditional token (CT) trading on the CTF exchange

Cost: ~0.01 POL in gas (one-time per wallet)

### 6. Verify Your Setup

```bash
# Check wallet balances
curl -s -X POST https://polygon-rpc.com -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "eth_getBalance",
  "params": ["YOUR_WALLET_ADDRESS", "latest"],
  "id": 1
}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'POL: {int(r[\"result\"], 16)/1e18:.4f}')"

# Check USDC.e balance
curl -s -X POST https://polygon-rpc.com -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "eth_call",
  "params": [{"to": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "data": "0x70a08231000000000000000000000000YOUR_ADDRESS_WITHOUT_0x"}, "latest"],
  "id": 1
}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'USDC.e: ${int(r[\"result\"], 16)/1e6:.2f}')"
```

### Common Funding Mistakes

| Mistake | Result | Solution |
|---------|--------|----------|
| Funded embedded wallet | Can't trade via API | Withdraw to EOA wallet |
| Sent native USDC | "Balance/allowance" error | Swap to USDC.e on DEX |
| No POL for gas | Transactions fail | Buy/transfer POL |
| Wrong network | Funds on Ethereum | Bridge to Polygon |
| No allowances set | "Not enough allowance" | Click Set Allowances |

## Security Checklist

- [ ] SSH key-only auth enabled
- [ ] Root login disabled
- [ ] UFW firewall enabled
- [ ] fail2ban running
- [ ] HTTPS only (no HTTP)
- [ ] Basic auth on all endpoints
- [ ] .env file with restricted permissions
- [ ] Non-root user running service
- [ ] Systemd sandboxing enabled
- [ ] Log rotation configured
- [ ] Trading wallet is a fresh EOA (not main wallet)
- [ ] Private key stored securely in .env
- [ ] POL balance sufficient for gas
- [ ] USDC.e (not native USDC) funded
- [ ] Token allowances set
