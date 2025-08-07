# Cloudflare Configuration for verteidiq.com PTaaS Platform

## DNS Records

### A Records
```
Type: A
Name: @
Content: YOUR_SERVER_IP
TTL: Auto
Proxy: Enabled (Orange Cloud)

Type: A  
Name: www
Content: YOUR_SERVER_IP
TTL: Auto
Proxy: Enabled (Orange Cloud)

Type: A
Name: api
Content: YOUR_SERVER_IP
TTL: Auto
Proxy: Enabled (Orange Cloud)
```

### CNAME Records
```
Type: CNAME
Name: ptaas
Target: verteidiq.com
TTL: Auto
Proxy: Enabled (Orange Cloud)

Type: CNAME
Name: portal
Target: verteidiq.com
TTL: Auto
Proxy: Enabled (Orange Cloud)

Type: CNAME
Name: customers
Target: verteidiq.com
TTL: Auto
Proxy: Enabled (Orange Cloud)
```

## SSL/TLS Configuration

### Encryption Mode
- **Mode**: Full (strict)
- **Edge Certificates**: Universal SSL enabled
- **Origin Server**: Use Origin CA certificates
- **Always Use HTTPS**: Enabled
- **HTTP Strict Transport Security (HSTS)**: Enabled
  - Max Age Header: 12 months
  - Include Subdomains: Yes
  - Preload: Yes
  - No-Sniff Header: Yes

### TLS Settings
- **Minimum TLS Version**: 1.2
- **TLS 1.3**: Enabled
- **Automatic HTTPS Rewrites**: Enabled
- **Certificate Transparency Monitoring**: Enabled

## Security Configuration

### WAF Rules
```javascript
// Block known attack patterns
(http.request.uri.path contains "/wp-admin" or 
 http.request.uri.path contains "/xmlrpc.php" or
 http.request.uri.path contains "/.env" or
 http.request.uri.path contains "/.git" or
 http.request.uri.path contains "/node_modules")

// Rate limiting for login endpoints
(http.request.uri.path contains "/api/auth" and http.request.method eq "POST")

// Block malicious user agents
(http.user_agent contains "sqlmap" or 
 http.user_agent contains "nikto" or
 http.user_agent contains "gobuster" or
 http.user_agent contains "dirbuster")

// Geographic restrictions (if needed)
(ip.geoip.country ne "US" and ip.geoip.country ne "DE" and ip.geoip.country ne "GB")
```

### Firewall Rules
1. **Challenge suspicious requests**
   - Expression: `(cf.threat_score gt 14)`
   - Action: Challenge (Captcha)

2. **Block high threat scores**
   - Expression: `(cf.threat_score gt 50)`
   - Action: Block

3. **Rate limit API endpoints**
   - Expression: `(http.request.uri.path matches "^/api/")`
   - Action: Rate limit (100 requests per minute)

4. **Rate limit authentication**
   - Expression: `(http.request.uri.path contains "/api/auth")`
   - Action: Rate limit (10 requests per minute)

### Bot Fight Mode
- **Bot Fight Mode**: Enabled
- **Super Bot Fight Mode**: Enabled (if available)
  - Definitely automated: Block
  - Likely automated: Challenge
  - Verified bots: Allow

## Performance Configuration

### Caching Rules

#### Static Assets
```javascript
// Rule 1: Cache static assets for 1 year
Expression: (http.request.uri.path matches "^.*\\.(css|js|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot|webp|avif)$")
Settings:
  - Cache Level: Cache Everything
  - Edge TTL: 1 year
  - Browser TTL: 1 year
```

#### Next.js Assets
```javascript
// Rule 2: Cache Next.js static files
Expression: (http.request.uri.path matches "^/_next/static/.*")
Settings:
  - Cache Level: Cache Everything
  - Edge TTL: 1 year
  - Browser TTL: 1 year
```

#### HTML Files
```javascript
// Rule 3: Short cache for HTML
Expression: (http.request.uri.path matches "^.*\\.html$" or http.request.uri.path eq "/")
Settings:
  - Cache Level: Cache Everything
  - Edge TTL: 1 hour
  - Browser TTL: 1 hour
```

#### API Routes
```javascript
// Rule 4: No cache for API routes
Expression: (http.request.uri.path matches "^/api/.*")
Settings:
  - Cache Level: Bypass
```

### Page Rules (Legacy - migrate to Rules Engine)

1. **Static Assets Caching**
   - URL: `verteidiq.com/_next/static/*`
   - Settings: Cache Level: Cache Everything, Edge Cache TTL: 1 year

2. **API Bypass**
   - URL: `verteidiq.com/api/*`
   - Settings: Cache Level: Bypass

3. **Main Domain Performance**
   - URL: `verteidiq.com/*`
   - Settings: Auto Minify: CSS, HTML, JS enabled

### Speed Optimizations
- **Auto Minify**: CSS, HTML, JavaScript enabled
- **Brotli**: Enabled
- **HTTP/2**: Enabled
- **HTTP/3 (QUIC)**: Enabled
- **0-RTT Connection Resumption**: Enabled
- **Rocket Loader**: Disabled (conflicts with React)
- **Mirage**: Enabled
- **Polish**: Enabled (Lossy)
- **WebP**: Enabled

## DDoS Protection

### DDoS Attack Protection
- **HTTP DDoS Attack Protection**: Enabled
- **Sensitivity Level**: Medium
- **Advanced DDoS Protection**: Enabled (if available)

### Rate Limiting
1. **Global Rate Limit**
   - Threshold: 1000 requests per minute per IP
   - Action: Challenge

2. **API Rate Limit**
   - Threshold: 100 requests per minute per IP for /api/*
   - Action: Block

3. **Login Rate Limit**
   - Threshold: 5 requests per minute per IP for /api/auth/*
   - Action: Block

## Analytics & Monitoring

### Analytics
- **Web Analytics**: Enabled
- **Zone Analytics**: Enabled
- **Security Analytics**: Enabled

### Monitoring
- **Load Balancing Health Checks**: Configure if using multiple origins
- **Origin Error Rate**: Monitor and alert
- **Cache Hit Rate**: Monitor (target >95%)

## Network Configuration

### Argo Smart Routing
- **Argo Smart Routing**: Enabled (if available)
- **Argo Tunnel**: Consider for origin protection

### Load Balancing (if needed)
```javascript
// Primary origin
{
  "name": "primary",
  "address": "YOUR_SERVER_IP",
  "enabled": true,
  "weight": 1
}

// Backup origin (if available)
{
  "name": "backup", 
  "address": "BACKUP_SERVER_IP",
  "enabled": true,
  "weight": 0
}
```

## Headers

### Security Headers (Already configured in Nginx, but as backup)
```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=()
```

### Custom Headers
```
X-Powered-By: XORB-PTaaS
X-Security-Level: Enterprise
Access-Control-Allow-Origin: https://verteidiq.com
```

## Workers (Advanced)

### Security Worker Script
```javascript
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  // Block suspicious paths
  const blockedPaths = ['/wp-admin', '/.env', '/.git', '/node_modules']
  if (blockedPaths.some(path => url.pathname.startsWith(path))) {
    return new Response('Forbidden', { status: 403 })
  }
  
  // Add security headers
  const response = await fetch(request)
  const newResponse = new Response(response.body, response)
  
  newResponse.headers.set('X-Security-Worker', 'Active')
  newResponse.headers.set('X-Timestamp', new Date().toISOString())
  
  return newResponse
}
```

## Implementation Checklist

### Initial Setup
- [ ] Add domain to Cloudflare
- [ ] Update nameservers at domain registrar
- [ ] Configure DNS records
- [ ] Set SSL/TLS to Full (strict)
- [ ] Generate Origin CA certificates
- [ ] Install Origin CA certificates on server

### Security Configuration
- [ ] Enable WAF with custom rules
- [ ] Configure Bot Fight Mode
- [ ] Set up rate limiting rules
- [ ] Configure firewall rules
- [ ] Enable DDoS protection

### Performance Configuration
- [ ] Set up caching rules
- [ ] Enable speed optimizations
- [ ] Configure compression settings
- [ ] Enable HTTP/2 and HTTP/3
- [ ] Set up Polish for images

### Monitoring & Analytics
- [ ] Enable Web Analytics
- [ ] Set up alerting for security events
- [ ] Monitor cache hit ratios
- [ ] Track Core Web Vitals
- [ ] Configure uptime monitoring

### Testing
- [ ] Test SSL configuration (SSLLabs.com)
- [ ] Verify caching is working correctly
- [ ] Test rate limiting functionality
- [ ] Verify security headers
- [ ] Performance test with Lighthouse
- [ ] Test failover scenarios (if applicable)