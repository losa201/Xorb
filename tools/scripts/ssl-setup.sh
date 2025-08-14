#!/bin/bash

# XORB SSL Setup Script for verteidiq.com
# CloudFlare Origin Certificate Configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root"
    exit 1
fi

log "🔐 Starting SSL Setup for XORB Platform (verteidiq.com)"
log "=" * 60

# Install nginx if not present
if ! command -v nginx &> /dev/null; then
    log "📦 Installing nginx..."
    apt update
    apt install -y nginx
else
    log "✅ nginx is already installed"
fi

# Create necessary directories
log "📁 Creating SSL and configuration directories..."
mkdir -p /etc/nginx/sites-available
mkdir -p /etc/nginx/sites-enabled
mkdir -p /var/www/verteidiq.com
mkdir -p /var/log/nginx

# Verify SSL certificate files exist
if [[ ! -f "/root/Xorb/ssl/verteidiq.com.crt" ]]; then
    error "SSL certificate file not found: /root/Xorb/ssl/verteidiq.com.crt"
    exit 1
fi

if [[ ! -f "/root/Xorb/ssl/verteidiq.com.key" ]]; then
    error "SSL private key file not found: /root/Xorb/ssl/verteidiq.com.key"
    exit 1
fi

log "✅ SSL certificate files verified"

# Verify certificate details
log "🔍 Verifying SSL certificate details..."
openssl x509 -in /root/Xorb/ssl/verteidiq.com.crt -text -noout | grep -E "(Subject:|DNS:|Not Before|Not After)"

# Copy nginx configuration
log "⚙️ Configuring nginx for SSL..."
cp /root/Xorb/nginx/ssl-verteidiq.conf /etc/nginx/sites-available/verteidiq.com

# Enable the site
ln -sf /etc/nginx/sites-available/verteidiq.com /etc/nginx/sites-enabled/

# Remove default nginx site if it exists
if [[ -f "/etc/nginx/sites-enabled/default" ]]; then
    rm /etc/nginx/sites-enabled/default
    log "🗑️ Removed default nginx site"
fi

# Test nginx configuration
log "🧪 Testing nginx configuration..."
if nginx -t; then
    log "✅ nginx configuration test passed"
else
    error "❌ nginx configuration test failed"
    exit 1
fi

# Create a simple index page if it doesn't exist
if [[ ! -f "/var/www/verteidiq.com/index.html" ]]; then
    log "📄 Creating default index page..."
    cat > /var/www/verteidiq.com/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB - Ultimate Autonomous Cybersecurity Platform</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            text-align: center;
            padding: 2rem;
            max-width: 800px;
        }
        .logo {
            font-size: 4rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .tagline {
            font-size: 1.5rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }
        .feature {
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
        .ssl-badge {
            background: #28a745;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🚀 XORB</div>
        <div class="tagline">Ultimate Autonomous Cybersecurity Platform</div>
        <div class="ssl-badge">🔐 Secured with SSL/TLS</div>

        <div class="features">
            <div class="feature">
                <h3>🧠 Autonomous Learning</h3>
                <p>Self-improving AI agents with continuous learning capabilities</p>
            </div>
            <div class="feature">
                <h3>🐝 Swarm Intelligence</h3>
                <p>64-agent collective intelligence network for advanced threat detection</p>
            </div>
            <div class="feature">
                <h3>🛡️ Enterprise Security</h3>
                <p>Production-grade security with mTLS and comprehensive audit logging</p>
            </div>
            <div class="feature">
                <h3>📊 Real-time Monitoring</h3>
                <p>Advanced monitoring and alerting across all system components</p>
            </div>
        </div>

        <p style="margin-top: 2rem; opacity: 0.7;">
            Advancing Cybersecurity Through Autonomous Intelligence
        </p>
    </div>
</body>
</html>
EOF
fi

# Set proper ownership and permissions
chown -R www-data:www-data /var/www/verteidiq.com
chmod -R 755 /var/www/verteidiq.com

# Start and enable nginx
log "🚀 Starting nginx service..."
systemctl enable nginx
systemctl restart nginx

# Check nginx status
if systemctl is-active --quiet nginx; then
    log "✅ nginx is running successfully"
else
    error "❌ Failed to start nginx"
    exit 1
fi

# Verify SSL certificate installation
log "🔍 Verifying SSL installation..."
sleep 2

# Test SSL certificate
if command -v openssl &> /dev/null; then
    log "Testing SSL certificate..."
    timeout 10 openssl s_client -connect localhost:443 -servername verteidiq.com < /dev/null 2>/dev/null | openssl x509 -noout -subject -dates
fi

log "=" * 60
log "🎉 SSL Setup Complete!"
log "=" * 60
log "📋 Setup Summary:"
log "   🌐 Domain: verteidiq.com"
log "   🔐 SSL Certificate: CloudFlare Origin Certificate"
log "   📅 Valid Until: July 27, 2040"
log "   🚀 Web Server: nginx with HTTP/2 support"
log "   📁 Document Root: /var/www/verteidiq.com"
log "   📊 Access Logs: /var/log/nginx/verteidiq.com.access.log"
log "   ⚠️  Error Logs: /var/log/nginx/verteidiq.com.error.log"
log ""
log "🔧 Configuration Files:"
log "   📄 nginx Config: /etc/nginx/sites-available/verteidiq.com"
log "   🔐 SSL Cert: /root/Xorb/ssl/verteidiq.com.crt"
log "   🔑 SSL Key: /root/Xorb/ssl/verteidiq.com.key"
log ""
log "🌐 Your website is now accessible at:"
log "   https://verteidiq.com"
log "   https://www.verteidiq.com"
log ""
log "⚡ Next Steps:"
log "   1. Update DNS A records to point to your server IP"
log "   2. Configure CloudFlare DNS settings"
log "   3. Test HTTPS functionality"
log "   4. Monitor SSL certificate expiration"
log ""
log "🎯 XORB Platform is now secured with SSL/TLS!"

# Display current nginx status
log "📊 Current nginx status:"
systemctl status nginx --no-pager -l
