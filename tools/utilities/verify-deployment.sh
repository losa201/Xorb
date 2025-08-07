#!/bin/bash

echo "🔍 PTaaS Frontend Deployment Verification"
echo "========================================"

# Test local Next.js server
echo "1. Testing Next.js server (localhost:3005)..."
if curl -s http://localhost:3005/api/health > /dev/null; then
    echo "   ✅ Next.js server is running"
    echo "   📊 Health: $(curl -s http://localhost:3005/api/health | jq -r '.status // "healthy"')"
else
    echo "   ❌ Next.js server is not responding"
fi

# Test Nginx status
echo ""
echo "2. Testing Nginx status..."
if systemctl is-active nginx > /dev/null; then
    echo "   ✅ Nginx is running"
else
    echo "   ❌ Nginx is not running"
fi

# Test domain configuration
echo ""
echo "3. Testing domain configuration..."
if grep -q "verteidiq.com" /etc/nginx/sites-enabled/ptaas-frontend.conf 2>/dev/null; then
    echo "   ✅ Domain configuration found"
else
    echo "   ❌ Domain configuration missing"
fi

# Test port availability
echo ""
echo "4. Testing port availability..."
if netstat -ln | grep -q ":3005 "; then
    echo "   ✅ Port 3005 is in use (Next.js)"
else
    echo "   ❌ Port 3005 is not in use"
fi

if netstat -ln | grep -q ":80 "; then
    echo "   ✅ Port 80 is in use (Nginx)"
else
    echo "   ❌ Port 80 is not in use"
fi

# Final status
echo ""
echo "🎯 Deployment Status:"
echo "   • Frontend: http://localhost:3005"
echo "   • Domain: http://verteidiq.com (configured)"
echo "   • Health: http://localhost:3005/api/health"
echo ""
echo "📝 Logs:"
echo "   • Nginx: journalctl -f -u nginx"
echo "   • Frontend: tail -f /var/log/ptaas-frontend.log"
echo ""
echo "✨ Deployment verification complete!"