# AWS Lightsail Deployment Guide

## Recommended Configuration

- **Region**: eu-west-1 (Ireland) - **Not geoblocked + close to Polymarket**
- **Plan**: $10/month (2 GB RAM, 2 vCPU, 60 GB SSD, 3 TB transfer)
- **OS**: Ubuntu 22.04 LTS

### Why Ireland?

Polymarket's CLOB servers are in AWS eu-west-2 (London), but **UK is geoblocked**.

Ireland (eu-west-1) is the best option:
- **Not geoblocked** - can place live trades
- **~5-15ms latency** to London via AWS backbone
- **Same pricing** as other EU regions

Geoblocked regions (cannot trade): US, UK, Germany, France, Italy, Poland, etc.
See: <https://docs.polymarket.com/polymarket-learn/FAQ/geoblocking>

### Region Comparison

| Region | Latency to Polymarket | Can Trade? |
|--------|----------------------|------------|
| eu-west-1 (Ireland) | ~5-15ms | Yes |
| eu-west-2 (London) | <5ms | No (blocked) |
| eu-central-1 (Frankfurt) | ~10-20ms | No (blocked) |

## 1. Create Lightsail Instance

### Via AWS Console

1. Go to [Lightsail Console](https://lightsail.aws.amazon.com)
2. Click "Create instance"
3. Select:
   - Region: Ireland (eu-west-1)
   - Platform: Linux/Unix
   - Blueprint: Ubuntu 22.04 LTS
   - Plan: $10 (2 GB RAM)
   - Name: `predmkt-prod`
4. Create instance

### Via AWS CLI

```bash
aws lightsail create-instances \
  --instance-names predmkt-prod \
  --availability-zone eu-west-1a \
  --blueprint-id ubuntu_22_04 \
  --bundle-id medium_3_0 \
  --key-pair-name your-key-pair
```

## 2. Network Configuration

### Attach Static IP

```bash
# Create static IP
aws lightsail allocate-static-ip --static-ip-name predmkt-ip

# Attach to instance
aws lightsail attach-static-ip \
  --static-ip-name predmkt-ip \
  --instance-name predmkt-prod
```

### Configure Firewall

In Lightsail console → Instance → Networking:

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP only | SSH |
| 443 | TCP | Anywhere | HTTPS |

**Remove** the default HTTP (80) rule after SSL is set up.

```bash
# Via CLI
aws lightsail put-instance-public-ports \
  --instance-name predmkt-prod \
  --port-infos \
    "fromPort=22,toPort=22,protocol=tcp,cidrs=[YOUR_IP/32]" \
    "fromPort=443,toPort=443,protocol=tcp"
```

## 3. Initial Server Setup

SSH into your instance:

```bash
ssh -i your-key.pem ubuntu@YOUR_STATIC_IP
```

### System Hardening

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Create application user
sudo adduser predmkt --disabled-password --gecos ""

# Install dependencies
sudo apt install -y \
  python3.11 python3.11-venv python3-pip \
  nginx certbot python3-certbot-nginx \
  fail2ban ufw git curl jq

# Configure fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# UFW is already handled by Lightsail firewall, but as backup:
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from YOUR_IP to any port 22
sudo ufw allow 443/tcp
sudo ufw enable
```

### SSH Hardening

```bash
# Edit SSH config
sudo nano /etc/ssh/sshd_config

# Ensure these settings:
# PermitRootLogin no
# PasswordAuthentication no
# PubkeyAuthentication yes
# MaxAuthTries 3

sudo systemctl restart sshd
```

## 4. Application Deployment

```bash
# Switch to predmkt user
sudo su - predmkt

# Create directories
mkdir -p /home/predmkt/app
cd /home/predmkt/app

# Clone or copy your code
# Option 1: Git clone (if public repo)
git clone https://github.com/yourusername/predmkt.git .

# Option 2: SCP from local machine
# (run from local): scp -r ./backend ubuntu@YOUR_IP:/tmp/predmkt
# (on server): cp -r /tmp/predmkt/* /home/predmkt/app/

# Setup Python environment
cd /home/predmkt/app/backend
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Create Environment File

```bash
# As predmkt user
cat > /home/predmkt/.env << 'EOF'
# Predmkt Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
LOG_LEVEL=INFO

# Future: Add API keys here when going live
# POLYMARKET_API_KEY=
# POLYMARKET_API_SECRET=
EOF

chmod 600 /home/predmkt/.env
```

## 5. Systemd Service

```bash
# As root/sudo
sudo nano /etc/systemd/system/predmkt.service
```

Paste:

```ini
[Unit]
Description=Predmkt Paper Trading Server
After=network.target

[Service]
Type=simple
User=predmkt
Group=predmkt
WorkingDirectory=/home/predmkt/app/backend
Environment="PATH=/home/predmkt/app/backend/venv/bin"
EnvironmentFile=/home/predmkt/.env
ExecStart=/home/predmkt/app/backend/venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable predmkt
sudo systemctl start predmkt
sudo systemctl status predmkt
```

## 6. Nginx + SSL

### Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/predmkt
```

Paste (replace `your-domain.com`):

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL will be configured by certbot

    # Basic auth
    auth_basic "Predmkt";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Health check - no auth
    location /health {
        auth_basic off;
        proxy_pass http://127.0.0.1:8000;
    }

    # WebSocket
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # API
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Create password file
sudo apt install apache2-utils -y
sudo htpasswd -c /etc/nginx/.htpasswd admin
# Enter a strong password

# Enable site
sudo ln -s /etc/nginx/sites-available/predmkt /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### Get SSL Certificate

First, point your domain DNS to the static IP, then:

```bash
sudo certbot --nginx -d your-domain.com
```

## 7. Automated Backups

### Enable Lightsail Snapshots

```bash
# Create snapshot
aws lightsail create-instance-snapshot \
  --instance-name predmkt-prod \
  --instance-snapshot-name predmkt-backup-$(date +%Y%m%d)

# Enable automatic snapshots (daily at 06:00 UTC)
aws lightsail enable-add-on \
  --resource-name predmkt-prod \
  --add-on-request "addOnType=AutoSnapshot,autoSnapshotAddOnRequest={snapshotTimeOfDay=06:00}"
```

Cost: ~$0.05/GB × 60 GB = **$3/month** for daily snapshots (7-day retention).

### Application State Backup

```bash
# Cron job to backup trading state to S3 (optional)
# First, create an S3 bucket and configure AWS CLI

crontab -e
# Add:
0 */6 * * * aws s3 cp /home/predmkt/app/backend/paper_trading_state.json s3://your-bucket/backups/$(date +\%Y\%m\%d_\%H\%M).json
```

## 8. Monitoring

### Health Check Script

```bash
sudo nano /home/predmkt/healthcheck.sh
```

```bash
#!/bin/bash
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
if [ "$RESPONSE" != "200" ]; then
    echo "$(date): Health check failed, restarting..."
    sudo systemctl restart predmkt
fi
```

```bash
sudo chmod +x /home/predmkt/healthcheck.sh
sudo crontab -e
# Add:
*/5 * * * * /home/predmkt/healthcheck.sh >> /var/log/predmkt-health.log 2>&1
```

### CloudWatch Agent (Optional)

For more detailed monitoring, install CloudWatch agent to track:
- CPU, memory, disk usage
- Custom metrics (trade count, P&L)
- Log aggregation

## 9. Security Checklist

- [ ] SSH key-only authentication
- [ ] Root login disabled
- [ ] Lightsail firewall configured (SSH restricted to your IP)
- [ ] fail2ban running
- [ ] HTTPS only with valid SSL
- [ ] HTTP Basic Auth on all endpoints
- [ ] .env file with 600 permissions
- [ ] Application runs as non-root user
- [ ] Automatic snapshots enabled
- [ ] Health monitoring active

## 10. Costs Summary

| Item | Monthly Cost |
|------|-------------|
| Lightsail $10 instance | $10.00 |
| Static IP | Free (with instance) |
| Auto snapshots (60 GB) | ~$3.00 |
| Data transfer | Included (3 TB) |
| **Total** | **~$13/month** |

## Upgrade Path

If you outgrow Lightsail:

1. Create a Lightsail snapshot
2. Export snapshot to EC2 AMI
3. Launch EC2 instance from AMI
4. Configure VPC, security groups, load balancer as needed

Note: This is a one-way migration - EC2 cannot be converted back to Lightsail.

## Troubleshooting

### Check service status
```bash
sudo systemctl status predmkt
sudo journalctl -u predmkt -f
```

### Check nginx
```bash
sudo nginx -t
sudo tail -f /var/log/nginx/error.log
```

### Check health endpoint
```bash
curl http://127.0.0.1:8000/health
```

### Restart everything
```bash
sudo systemctl restart predmkt nginx
```
