#!/usr/bin/env python3
"""
XORB PTaaS Web Dashboard Service
Simple web dashboard with authentication
"""

import json
import time
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="XORB PTaaS Dashboard", version="1.0.0")

# Simple session storage (in production, use Redis or database)
sessions = {}

# Authentication credentials
VALID_CREDENTIALS = {
    "admin": "xorb_admin_2025",
    "demo": "demo123",
    "user": "user123"
}

class SessionManager:
    @staticmethod
    def create_session(username: str) -> str:
        session_id = f"session_{username}_{int(time.time())}"
        sessions[session_id] = {
            "username": username,
            "created_at": datetime.now(),
            "last_active": datetime.now()
        }
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        if session_id in sessions:
            sessions[session_id]["last_active"] = datetime.now()
            return sessions[session_id]
        return None
    
    @staticmethod
    def delete_session(session_id: str):
        if session_id in sessions:
            del sessions[session_id]

def get_current_user(request: Request) -> Optional[str]:
    session_id = request.cookies.get("session_id")
    if session_id:
        session = SessionManager.get_session(session_id)
        if session:
            return session["username"]
    return None

def require_auth(request: Request) -> str:
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB PTaaS Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        .logo {{ font-size: 1.5rem; font-weight: bold; color: white; }}
        .user-info {{ color: white; opacity: 0.9; }}
        
        .container {{
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }}
        
        .welcome-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 0.5rem;
        }}
        
        .stat-label {{ opacity: 0.8; }}
        
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        
        .feature-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 1.5rem;
            transition: transform 0.2s ease;
        }}
        
        .feature-card:hover {{
            transform: translateY(-2px);
            border-color: #667eea;
        }}
        
        .feature-icon {{
            font-size: 2rem;
            margin-bottom: 1rem;
        }}
        
        .feature-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #667eea;
        }}
        
        .btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: transform 0.2s ease;
        }}
        
        .btn:hover {{
            transform: translateY(-1px);
        }}
        
        .logout-btn {{
            background: rgba(239, 68, 68, 0.8);
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            margin-right: 0.5rem;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🛡️ XORB PTaaS Platform</div>
        <div class="user-info">
            <span class="status-indicator"></span>
            Welcome, {user} | 
            <a href="/logout" class="logout-btn btn">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="welcome-card">
            <h1>🚀 Welcome to XORB PTaaS</h1>
            <p style="margin-top: 1rem; opacity: 0.8;">
                Next-generation AI-powered Penetration Testing as a Service platform. 
                Your security testing and threat intelligence hub is ready for action.
            </p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">3</div>
                <div class="stat-label">Active Tenants</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">27</div>
                <div class="stat-label">Security Findings</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">15</div>
                <div class="stat-label">AI Insights</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">13ms</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
        </div>
        
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">ARIA AI Assistant</div>
                <p style="opacity: 0.8; margin-bottom: 1rem;">
                    Natural language security analysis with voice commands and contextual insights.
                </p>
                <a href="/api/v1/ai/chat" class="btn">Launch ARIA</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Security Scanning</div>
                <p style="opacity: 0.8; margin-bottom: 1rem;">
                    Advanced vulnerability assessment with real-time threat detection.
                </p>
                <a href="/api/v1/scans" class="btn">Start Scan</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Threat Intelligence</div>
                <p style="opacity: 0.8; margin-bottom: 1rem;">
                    Real-time threat landscape analysis and predictive security insights.
                </p>
                <a href="/api/v1/dashboard" class="btn">View Intelligence</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">👥</div>
                <div class="feature-title">Multi-Tenant Management</div>
                <p style="opacity: 0.8; margin-bottom: 1rem;">
                    Complete organizational security with isolated data and reporting.
                </p>
                <a href="/api/v1/tenants" class="btn">Manage Tenants</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Professional Reports</div>
                <p style="opacity: 0.8; margin-bottom: 1rem;">
                    AI-generated security reports with executive summaries and technical details.
                </p>
                <a href="/api/v1/reports" class="btn">Generate Report</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">Progressive Web App</div>
                <p style="opacity: 0.8; margin-bottom: 1rem;">
                    Install on any device with offline support and push notifications.
                </p>
                <button onclick="installPWA()" class="btn">Install App</button>
            </div>
        </div>
        
        <div style="margin-top: 3rem; text-align: center; opacity: 0.6;">
            <p>🔐 Platform Status: Operational | API: http://188.245.101.102:8001 | Version: 1.0.0</p>
        </div>
    </div>
    
    <script>
        // PWA installation
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {{
            e.preventDefault();
            deferredPrompt = e;
        }});
        
        function installPWA() {{
            if (deferredPrompt) {{
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {{
                    if (choiceResult.outcome === 'accepted') {{
                        console.log('PWA installed');
                    }}
                    deferredPrompt = null;
                }});
            }} else {{
                alert('PWA installation not available. Try accessing from a supported browser.');
            }}
        }}
        
        // Auto-refresh stats every 30 seconds
        setInterval(() => {{
            // In production, fetch real data
            console.log('Refreshing dashboard data...');
        }}, 30000);
    </script>
</body>
</html>
    """)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=302)
    
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB PTaaS Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: #e2e8f0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .login-container {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 3rem;
            width: 100%;
            max-width: 400px;
            backdrop-filter: blur(20px);
            box-shadow: 0 8px 40px rgba(0,0,0,0.3);
        }
        
        .logo {
            text-align: center;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .title {
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            text-align: center;
            opacity: 0.7;
            margin-bottom: 2rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .form-input {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
            font-size: 1rem;
            transition: border-color 0.2s ease;
        }
        
        .form-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .login-btn {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .login-btn:hover {
            transform: translateY(-1px);
        }
        
        .credentials-info {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 6px;
            font-size: 0.9rem;
        }
        
        .error-message {
            color: #ef4444;
            margin-top: 1rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">🛡️</div>
        <h1 class="title">XORB PTaaS</h1>
        <p class="subtitle">AI-Powered Security Platform</p>
        
        <form method="post" action="/login">
            <div class="form-group">
                <label class="form-label" for="username">Username</label>
                <input type="text" id="username" name="username" class="form-input" required>
            </div>
            
            <div class="form-group">
                <label class="form-label" for="password">Password</label>
                <input type="password" id="password" name="password" class="form-input" required>
            </div>
            
            <button type="submit" class="login-btn">Login to Dashboard</button>
        </form>
        
        <div class="credentials-info">
            <strong>Valid Credentials:</strong><br>
            • Username: <code>admin</code> | Password: <code>xorb_admin_2025</code><br>
            • Username: <code>demo</code> | Password: <code>demo123</code><br>
            • Username: <code>user</code> | Password: <code>user123</code>
        </div>
    </div>
</body>
</html>
    """)

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
        session_id = SessionManager.create_session(username)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=3600)
        return response
    else:
        return HTMLResponse(content="""
        <script>
            alert('Invalid credentials. Please try again.');
            window.location.href = '/login';
        </script>
        """)

@app.get("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        SessionManager.delete_session(session_id)
    
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="session_id")
    return response

@app.get("/api/status")
async def api_status():
    return {
        "status": "healthy",
        "service": "XORB PTaaS Dashboard",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "dashboard"}

if __name__ == "__main__":
    print("🚀 Starting XORB PTaaS Web Dashboard...")
    print("📱 Dashboard URL: http://188.245.101.102:3005")
    print("🔐 Credentials: admin/xorb_admin_2025, demo/demo123, user/user123")
    uvicorn.run(app, host="0.0.0.0", port=3005)