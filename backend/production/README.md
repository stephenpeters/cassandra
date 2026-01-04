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
