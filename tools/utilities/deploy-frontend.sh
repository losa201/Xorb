#!/bin/bash

# Deploy PTaaS Frontend to verteidiq.com
set -e

echo "🚀 Deploying PTaaS Frontend to verteidiq.com..."

# Change to frontend directory
cd /root/Xorb/ptaas-frontend

# Stop any existing processes on port 3005
echo "🔄 Stopping existing processes..."
pkill -f "next start" || true
sleep 2

# Start the Next.js production server
echo "▶️  Starting Next.js production server on port 3005..."
export PORT=3005
nohup npm run start > /var/log/ptaas-frontend.log 2>&1 &
sleep 3

# Check if the server started successfully
if curl -f http://localhost:3005/api/health > /dev/null 2>&1; then
    echo "✅ Frontend server started successfully on port 3005"
else
    echo "❌ Failed to start frontend server"
    exit 1
fi

# Create a simple Nginx configuration for the domain
echo "🔧 Creating Nginx configuration..."
cat > /tmp/ptaas-frontend.conf << 'EOF'
server {
    listen 80;
    server_name verteidiq.com www.verteidiq.com;

    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    location / {
        proxy_pass http://localhost:3005;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

echo "📋 Nginx configuration created at /tmp/ptaas-frontend.conf"
echo "📝 To activate, copy to /etc/nginx/sites-available/ and enable"

echo "🎉 PTaaS Frontend deployment completed!"
echo "🌐 Application is running on:"
echo "   - Local: http://localhost:3005"
echo "   - Domain: http://verteidiq.com (after Nginx configuration)"
echo ""
echo "📊 Monitor logs with: tail -f /var/log/ptaas-frontend.log"
echo "🔍 Check status with: curl http://localhost:3005/health"
