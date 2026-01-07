#!/bin/bash
set -e

# Predmkt Production Deployment Script
# Run as: sudo ./deploy.sh

DOMAIN="${DOMAIN:-your-domain.com}"
EMAIL="${EMAIL:-admin@your-domain.com}"

echo "=== Predmkt Production Deployment ==="
echo "Domain: $DOMAIN"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./deploy.sh)"
    exit 1
fi

# System updates
echo "=== Updating system ==="
apt update && apt upgrade -y

# Create user if not exists
if ! id "predmkt" &>/dev/null; then
    echo "=== Creating predmkt user ==="
    useradd -m -s /bin/bash predmkt
fi

# Install dependencies
echo "=== Installing dependencies ==="
apt install -y \
    python3.11 python3.11-venv python3-pip \
    nginx certbot python3-certbot-nginx \
    apache2-utils fail2ban ufw \
    git curl jq

# Install Node.js 20 LTS for frontend build
echo "=== Installing Node.js ==="
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
fi
echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"

# Firewall setup
echo "=== Configuring firewall ==="
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Setup application directory
echo "=== Setting up application ==="
mkdir -p /opt/predmkt
mkdir -p /var/log/predmkt

if [ -d "/opt/predmkt/backend" ]; then
    echo "Updating existing installation..."
    cd /opt/predmkt
    git pull || true
else
    echo "Cloning repository..."
    cd /opt
    # Replace with your actual repo URL
    git clone . predmkt || {
        echo "No git repo available, copying local files..."
        mkdir -p /opt/predmkt/backend
    }
fi

# Copy production files (backend)
cp -r "$(dirname "$0")"/../* /opt/predmkt/backend/ 2>/dev/null || true

# Copy frontend if it exists in parent directory
SCRIPT_DIR="$(dirname "$0")"
if [ -d "$SCRIPT_DIR/../../frontend" ]; then
    echo "Copying frontend files..."
    cp -r "$SCRIPT_DIR/../../frontend" /opt/predmkt/
fi

# Set permissions
chown -R predmkt:predmkt /opt/predmkt
chown -R predmkt:predmkt /var/log/predmkt

# Python environment
echo "=== Setting up Python environment ==="
cd /opt/predmkt/backend
sudo -u predmkt python3.11 -m venv venv
sudo -u predmkt ./venv/bin/pip install --upgrade pip
sudo -u predmkt ./venv/bin/pip install -r requirements.txt

# Create .env file if not exists
if [ ! -f "/opt/predmkt/.env" ]; then
    echo "=== Creating .env file ==="

    # Generate a secure API key
    GENERATED_API_KEY=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)

    cat > /opt/predmkt/.env << EOF
# Predmkt Environment Configuration
# Keep this file secure!

# Environment (production requires API_KEY to be set)
ENV=production

# API Authentication (REQUIRED in production)
# This key is required for protected endpoints
API_KEY=${GENERATED_API_KEY}

# Server settings
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
LOG_LEVEL=INFO

# Telegram Alerts (optional)
# TELEGRAM_BOT_TOKEN=
# TELEGRAM_CHAT_ID=

# Live Trading Keys (add when ready for live trading)
# POLYMARKET_API_KEY=
# POLYMARKET_API_SECRET=
# POLYMARKET_PASSPHRASE=
EOF
    chmod 600 /opt/predmkt/.env
    chown predmkt:predmkt /opt/predmkt/.env

    echo ""
    echo "=== IMPORTANT: API Key Generated ==="
    echo "Your API key has been generated and saved to /opt/predmkt/.env"
    echo "API_KEY: ${GENERATED_API_KEY}"
    echo "Save this key securely - you'll need it for protected API endpoints."
    echo ""
fi

# Build and deploy frontend
echo "=== Building frontend ==="
mkdir -p /var/www/predmkt

if [ -d "/opt/predmkt/frontend" ]; then
    cd /opt/predmkt/frontend

    # Install dependencies
    npm ci --production=false

    # Build with production URLs
    NEXT_PUBLIC_WS_URL="wss://$DOMAIN/ws" \
    NEXT_PUBLIC_API_URL="https://$DOMAIN" \
    npm run build

    # Deploy static files
    rm -rf /var/www/predmkt/*
    cp -r out/* /var/www/predmkt/

    # Set permissions
    chown -R www-data:www-data /var/www/predmkt
    chmod -R 755 /var/www/predmkt

    echo "Frontend deployed to /var/www/predmkt"
else
    echo "Warning: Frontend directory not found at /opt/predmkt/frontend"
    echo "You can manually deploy the frontend later"
fi

# Setup systemd service
echo "=== Installing systemd service ==="
cp /opt/predmkt/backend/production/predmkt.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable predmkt

# Setup nginx
echo "=== Configuring nginx ==="
# Create htpasswd if not exists
if [ ! -f "/etc/nginx/.htpasswd" ]; then
    echo "Creating HTTP Basic Auth password..."
    echo "Enter password for 'admin' user:"
    htpasswd -c /etc/nginx/.htpasswd admin
fi

# Copy nginx config
cp /opt/predmkt/backend/production/nginx.conf /etc/nginx/sites-available/predmkt
sed -i "s/your-domain.com/$DOMAIN/g" /etc/nginx/sites-available/predmkt
ln -sf /etc/nginx/sites-available/predmkt /etc/nginx/sites-enabled/

# Remove default site
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# Get SSL certificate
echo "=== Obtaining SSL certificate ==="
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" || {
    echo "SSL setup failed - continuing without SSL for now"
    echo "Run: certbot --nginx -d $DOMAIN"
}

# Start services
echo "=== Starting services ==="
systemctl restart nginx
systemctl start predmkt

# Setup health check cron
echo "=== Setting up health check ==="
cat > /opt/predmkt/healthcheck.sh << 'EOF'
#!/bin/bash
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
if [ "$RESPONSE" != "200" ]; then
    echo "$(date): Health check failed (HTTP $RESPONSE), restarting..."
    systemctl restart predmkt
fi
EOF
chmod +x /opt/predmkt/healthcheck.sh
chown predmkt:predmkt /opt/predmkt/healthcheck.sh

# Add to crontab
(crontab -l 2>/dev/null || true; echo "*/5 * * * * /opt/predmkt/healthcheck.sh >> /var/log/predmkt/healthcheck.log 2>&1") | crontab -

# Setup log rotation
cat > /etc/logrotate.d/predmkt << 'EOF'
/var/log/predmkt/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 predmkt predmkt
}
EOF

# Setup S3 backups (optional - requires AWS CLI and credentials)
echo "=== Setting up S3 backups ==="

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    apt install -y awscli
fi

# Create backup script
cat > /opt/predmkt/backup-to-s3.sh << 'BACKUP_EOF'
#!/bin/bash
# Predmkt S3 Backup Script
# Backs up trades.db and state files to S3

set -e

# Configuration (set these in /opt/predmkt/.env)
source /opt/predmkt/.env

if [ -z "$S3_BACKUP_BUCKET" ]; then
    echo "$(date): S3_BACKUP_BUCKET not set, skipping backup"
    exit 0
fi

S3_PREFIX="${S3_BACKUP_PREFIX:-predmkt/backups}"
BACKUP_DIR="/opt/predmkt/backend"
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "$(date): Starting backup to s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}"

# Create local backup directory
mkdir -p /tmp/predmkt-backup

# Backup trades.db (with lock to prevent corruption)
if [ -f "${BACKUP_DIR}/trades.db" ]; then
    sqlite3 "${BACKUP_DIR}/trades.db" ".backup '/tmp/predmkt-backup/trades_${TIMESTAMP}.db'"
    gzip -f /tmp/predmkt-backup/trades_${TIMESTAMP}.db
    aws s3 cp /tmp/predmkt-backup/trades_${TIMESTAMP}.db.gz \
        "s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/${DATE}/trades_${TIMESTAMP}.db.gz"
    echo "$(date): Backed up trades.db"
fi

# Backup state files
for state_file in paper_trading_state.json historical_markets.json; do
    if [ -f "${BACKUP_DIR}/${state_file}" ]; then
        cp "${BACKUP_DIR}/${state_file}" /tmp/predmkt-backup/
        gzip -f /tmp/predmkt-backup/${state_file}
        aws s3 cp /tmp/predmkt-backup/${state_file}.gz \
            "s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/${DATE}/${state_file}.gz"
        echo "$(date): Backed up ${state_file}"
    fi
done

# Backup .env (encrypted if possible)
if [ -f "/opt/predmkt/.env" ]; then
    cp /opt/predmkt/.env /tmp/predmkt-backup/env_backup
    gzip -f /tmp/predmkt-backup/env_backup
    aws s3 cp /tmp/predmkt-backup/env_backup.gz \
        "s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/${DATE}/env_${TIMESTAMP}.gz" \
        --sse AES256
    echo "$(date): Backed up .env (encrypted)"
fi

# Cleanup old backups (keep last 7 days)
aws s3 ls "s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/" | while read -r line; do
    dir_date=$(echo "$line" | awk '{print $2}' | tr -d '/')
    if [[ "$dir_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        days_old=$(( ($(date +%s) - $(date -d "$dir_date" +%s)) / 86400 ))
        if [ "$days_old" -gt 7 ]; then
            echo "$(date): Removing old backup: $dir_date ($days_old days old)"
            aws s3 rm "s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/${dir_date}/" --recursive
        fi
    fi
done

# Cleanup local temp files
rm -rf /tmp/predmkt-backup

echo "$(date): Backup complete"
BACKUP_EOF

chmod +x /opt/predmkt/backup-to-s3.sh
chown predmkt:predmkt /opt/predmkt/backup-to-s3.sh

# Add S3 backup to crontab (runs daily at 3 AM)
(crontab -l 2>/dev/null || true; echo "0 3 * * * /opt/predmkt/backup-to-s3.sh >> /var/log/predmkt/backup.log 2>&1") | sort -u | crontab -

echo ""
echo "S3 backup configured. To enable:"
echo "  1. Add to /opt/predmkt/.env:"
echo "     S3_BACKUP_BUCKET=your-bucket-name"
echo "     S3_BACKUP_PREFIX=predmkt/backups"
echo "  2. Configure AWS credentials for the predmkt user:"
echo "     sudo -u predmkt aws configure"
echo ""

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Services:"
echo "  - Backend: systemctl status predmkt"
echo "  - Nginx:   systemctl status nginx"
echo ""
echo "Logs:"
echo "  - Backend: journalctl -u predmkt -f"
echo "  - Nginx:   tail -f /var/log/nginx/predmkt_*.log"
echo ""
echo "Health check: curl https://$DOMAIN/health"
echo ""
echo "IMPORTANT: Review /opt/predmkt/.env and add any API keys needed"
echo ""
