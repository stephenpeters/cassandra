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

# Copy production files
cp -r "$(dirname "$0")"/../* /opt/predmkt/backend/ 2>/dev/null || true

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
    cat > /opt/predmkt/.env << 'EOF'
# Predmkt Environment Configuration
# Keep this file secure!

# Server settings
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
LOG_LEVEL=INFO

# API Keys (add when needed for live trading)
# POLYMARKET_API_KEY=
# POLYMARKET_API_SECRET=
EOF
    chmod 600 /opt/predmkt/.env
    chown predmkt:predmkt /opt/predmkt/.env
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
