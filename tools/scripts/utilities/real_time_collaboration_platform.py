#!/usr/bin/env python3
"""
XORB Real-time Collaboration Platform
Multi-analyst collaboration for threat intelligence and incident response
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB Real-time Collaboration Platform",
    description="Multi-analyst collaboration for cybersecurity operations",
    version="4.0.0"
)

class MessageType(str, Enum):
    CHAT = "chat"
    ALERT = "alert"
    ANNOTATION = "annotation"
    STATUS_CHANGE = "status_change"
    THREAT_UPDATE = "threat_update"
    INCIDENT_UPDATE = "incident_update"
    COLLABORATION_REQUEST = "collaboration_request"

class UserRole(str, Enum):
    ANALYST = "analyst"
    SENIOR_ANALYST = "senior_analyst"
    SOC_MANAGER = "soc_manager"
    INCIDENT_COMMANDER = "incident_commander"
    THREAT_HUNTER = "threat_hunter"

class SessionStatus(str, Enum):
    ACTIVE = "active"
    INVESTIGATION = "investigation"
    INCIDENT_RESPONSE = "incident_response"
    BRIEFING = "briefing"
    IDLE = "idle"

@dataclass
class CollaborationUser:
    user_id: str
    username: str
    role: UserRole
    status: str
    last_activity: datetime
    current_session: Optional[str] = None
    permissions: List[str] = None

class CollaborationMessage(BaseModel):
    message_id: str
    user_id: str
    username: str
    message_type: MessageType
    content: str
    session_id: str
    timestamp: str
    metadata: Optional[Dict] = None
    mentions: List[str] = []
    attachments: List[str] = []

class CollaborationSession(BaseModel):
    session_id: str
    session_name: str
    session_type: str
    status: SessionStatus
    participants: List[str]
    created_by: str
    created_at: str
    last_activity: str
    incident_id: Optional[str] = None
    threat_indicators: List[str] = []
    shared_resources: Dict = {}

class ThreatAnnotation(BaseModel):
    annotation_id: str
    user_id: str
    username: str
    indicator_value: str
    annotation_type: str
    content: str
    confidence: float
    timestamp: str
    session_id: str

class IncidentUpdate(BaseModel):
    update_id: str
    incident_id: str
    user_id: str
    username: str
    update_type: str
    content: str
    severity_change: Optional[str] = None
    status_change: Optional[str] = None
    timestamp: str

class CollaborationConnectionManager:
    """Advanced WebSocket connection manager for collaboration"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.session_participants: Dict[str, Set[str]] = {}  # session_id -> set of user_ids
        
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str = None):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
        if session_id:
            self.user_sessions[user_id] = session_id
            if session_id not in self.session_participants:
                self.session_participants[session_id] = set()
            self.session_participants[session_id].add(user_id)
    
    def disconnect(self, user_id: str):
        # Remove from session participants
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            if session_id in self.session_participants:
                self.session_participants[session_id].discard(user_id)
                if not self.session_participants[session_id]:
                    del self.session_participants[session_id]
            del self.user_sessions[user_id]
        
        # Remove connection
        if user_id in self.active_connections:
            del self.active_connections[user_id]
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message)
            except:
                self.disconnect(user_id)
    
    async def broadcast_to_session(self, message: str, session_id: str, exclude_user: str = None):
        if session_id in self.session_participants:
            for user_id in self.session_participants[session_id]:
                if user_id != exclude_user:
                    await self.send_personal_message(message, user_id)
    
    async def broadcast_to_all(self, message: str):
        for user_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, user_id)

class CollaborationPlatform:
    """Advanced collaboration platform for cybersecurity operations"""
    
    def __init__(self):
        self.connection_manager = CollaborationConnectionManager()
        self.active_users: Dict[str, CollaborationUser] = {}
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        self.message_history: Dict[str, List[CollaborationMessage]] = {}
        self.threat_annotations: List[ThreatAnnotation] = []
        self.incident_updates: List[IncidentUpdate] = []
        
        # Predefined user roles and permissions
        self.role_permissions = {
            UserRole.ANALYST: ["view_threats", "annotate", "chat", "create_annotations"],
            UserRole.SENIOR_ANALYST: ["view_threats", "annotate", "chat", "create_annotations", "manage_incidents", "review_analysis"],
            UserRole.SOC_MANAGER: ["view_threats", "annotate", "chat", "create_annotations", "manage_incidents", "review_analysis", "manage_sessions", "assign_tasks"],
            UserRole.INCIDENT_COMMANDER: ["all_permissions"],
            UserRole.THREAT_HUNTER: ["view_threats", "annotate", "chat", "create_annotations", "deep_analysis", "create_hunts"]
        }
        
    def add_user(self, user_id: str, username: str, role: UserRole) -> CollaborationUser:
        """Add user to collaboration platform"""
        permissions = self.role_permissions.get(role, [])
        user = CollaborationUser(
            user_id=user_id,
            username=username,
            role=role,
            status="online",
            last_activity=datetime.now(),
            permissions=permissions
        )
        self.active_users[user_id] = user
        return user
    
    def create_collaboration_session(self, session_name: str, session_type: str, created_by: str, incident_id: str = None) -> CollaborationSession:
        """Create new collaboration session"""
        session_id = f"session_{int(time.time())}_{len(self.collaboration_sessions)}"
        
        session = CollaborationSession(
            session_id=session_id,
            session_name=session_name,
            session_type=session_type,
            status=SessionStatus.ACTIVE,
            participants=[created_by],
            created_by=created_by,
            created_at=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat(),
            incident_id=incident_id,
            threat_indicators=[],
            shared_resources={}
        )
        
        self.collaboration_sessions[session_id] = session
        self.message_history[session_id] = []
        return session
    
    async def add_message(self, user_id: str, session_id: str, content: str, message_type: MessageType, metadata: Dict = None) -> CollaborationMessage:
        """Add message to collaboration session"""
        if session_id not in self.collaboration_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        user = self.active_users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        message = CollaborationMessage(
            message_id=f"msg_{int(time.time())}_{len(self.message_history[session_id])}",
            user_id=user_id,
            username=user.username,
            message_type=message_type,
            content=content,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            mentions=[],
            attachments=[]
        )
        
        # Add to message history
        self.message_history[session_id].append(message)
        
        # Update session activity
        self.collaboration_sessions[session_id].last_activity = datetime.now().isoformat()
        
        # Broadcast to session participants
        await self.connection_manager.broadcast_to_session(
            json.dumps(message.dict()),
            session_id,
            exclude_user=user_id
        )
        
        return message
    
    async def add_threat_annotation(self, user_id: str, session_id: str, indicator_value: str, annotation_type: str, content: str, confidence: float) -> ThreatAnnotation:
        """Add threat annotation"""
        user = self.active_users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        annotation = ThreatAnnotation(
            annotation_id=f"anno_{int(time.time())}_{len(self.threat_annotations)}",
            user_id=user_id,
            username=user.username,
            indicator_value=indicator_value,
            annotation_type=annotation_type,
            content=content,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
        self.threat_annotations.append(annotation)
        
        # Broadcast annotation to session
        annotation_message = {
            "type": "threat_annotation",
            "data": annotation.dict()
        }
        
        await self.connection_manager.broadcast_to_session(
            json.dumps(annotation_message),
            session_id
        )
        
        return annotation
    
    async def update_incident(self, user_id: str, incident_id: str, update_type: str, content: str, severity_change: str = None, status_change: str = None) -> IncidentUpdate:
        """Add incident update"""
        user = self.active_users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        update = IncidentUpdate(
            update_id=f"inc_update_{int(time.time())}_{len(self.incident_updates)}",
            incident_id=incident_id,
            user_id=user_id,
            username=user.username,
            update_type=update_type,
            content=content,
            severity_change=severity_change,
            status_change=status_change,
            timestamp=datetime.now().isoformat()
        )
        
        self.incident_updates.append(update)
        
        # Broadcast to all relevant sessions
        incident_message = {
            "type": "incident_update",
            "data": update.dict()
        }
        
        # Find sessions related to this incident
        for session in self.collaboration_sessions.values():
            if session.incident_id == incident_id:
                await self.connection_manager.broadcast_to_session(
                    json.dumps(incident_message),
                    session.session_id
                )
        
        return update
    
    def get_session_analytics(self, session_id: str) -> Dict:
        """Get analytics for collaboration session"""
        if session_id not in self.collaboration_sessions:
            return {}
        
        session = self.collaboration_sessions[session_id]
        messages = self.message_history.get(session_id, [])
        
        # Message analytics
        message_types = {}
        user_activity = {}
        
        for msg in messages:
            message_types[msg.message_type] = message_types.get(msg.message_type, 0) + 1
            user_activity[msg.username] = user_activity.get(msg.username, 0) + 1
        
        # Time-based activity
        last_hour_messages = len([
            msg for msg in messages 
            if datetime.fromisoformat(msg.timestamp) > datetime.now() - timedelta(hours=1)
        ])
        
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "message_types": message_types,
            "user_activity": user_activity,
            "active_participants": len(session.participants),
            "last_hour_activity": last_hour_messages,
            "session_duration": str(datetime.now() - datetime.fromisoformat(session.created_at)),
            "threat_annotations": len([
                ann for ann in self.threat_annotations 
                if ann.session_id == session_id
            ])
        }

# Initialize collaboration platform
collaboration_platform = CollaborationPlatform()

# Add sample users
collaboration_platform.add_user("user_analyst_01", "Alice_Chen", UserRole.SENIOR_ANALYST)
collaboration_platform.add_user("user_analyst_02", "Bob_Rodriguez", UserRole.ANALYST)
collaboration_platform.add_user("user_manager_01", "Carol_Singh", UserRole.SOC_MANAGER)
collaboration_platform.add_user("user_hunter_01", "David_Kim", UserRole.THREAT_HUNTER)
collaboration_platform.add_user("user_commander_01", "Eve_Johnson", UserRole.INCIDENT_COMMANDER)

@app.websocket("/ws/collaboration/{user_id}")
async def websocket_collaboration(websocket: WebSocket, user_id: str, session_id: str = None):
    """WebSocket endpoint for real-time collaboration"""
    await collaboration_platform.connection_manager.connect(websocket, user_id, session_id)
    
    # Send welcome message
    welcome_message = {
        "type": "connection_established",
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }
    await websocket.send_text(json.dumps(welcome_message))
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("action") == "send_message":
                await collaboration_platform.add_message(
                    user_id=user_id,
                    session_id=message_data["session_id"],
                    content=message_data["content"],
                    message_type=MessageType(message_data.get("message_type", "chat")),
                    metadata=message_data.get("metadata")
                )
            
            elif message_data.get("action") == "add_annotation":
                await collaboration_platform.add_threat_annotation(
                    user_id=user_id,
                    session_id=message_data["session_id"],
                    indicator_value=message_data["indicator_value"],
                    annotation_type=message_data["annotation_type"],
                    content=message_data["content"],
                    confidence=message_data["confidence"]
                )
            
            elif message_data.get("action") == "heartbeat":
                # Update user activity
                if user_id in collaboration_platform.active_users:
                    collaboration_platform.active_users[user_id].last_activity = datetime.now()
                
    except WebSocketDisconnect:
        collaboration_platform.connection_manager.disconnect(user_id)

@app.post("/collaboration/sessions")
async def create_session(
    session_name: str,
    session_type: str,
    created_by: str,
    incident_id: str = None
):
    """Create new collaboration session"""
    session = collaboration_platform.create_collaboration_session(
        session_name=session_name,
        session_type=session_type,
        created_by=created_by,
        incident_id=incident_id
    )
    return session.dict()

@app.get("/collaboration/sessions")
async def get_sessions():
    """Get all collaboration sessions"""
    return {
        "total_sessions": len(collaboration_platform.collaboration_sessions),
        "sessions": [session.dict() for session in collaboration_platform.collaboration_sessions.values()]
    }

@app.get("/collaboration/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific collaboration session"""
    if session_id not in collaboration_platform.collaboration_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = collaboration_platform.collaboration_sessions[session_id]
    messages = collaboration_platform.message_history.get(session_id, [])
    analytics = collaboration_platform.get_session_analytics(session_id)
    
    return {
        "session": session.dict(),
        "message_count": len(messages),
        "recent_messages": [msg.dict() for msg in messages[-20:]],  # Last 20 messages
        "analytics": analytics
    }

@app.get("/collaboration/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 100):
    """Get messages for collaboration session"""
    if session_id not in collaboration_platform.message_history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = collaboration_platform.message_history[session_id]
    return {
        "session_id": session_id,
        "total_messages": len(messages),
        "messages": [msg.dict() for msg in messages[-limit:]]
    }

@app.get("/collaboration/annotations")
async def get_threat_annotations(session_id: str = None, indicator_value: str = None):
    """Get threat annotations"""
    annotations = collaboration_platform.threat_annotations
    
    if session_id:
        annotations = [ann for ann in annotations if ann.session_id == session_id]
    
    if indicator_value:
        annotations = [ann for ann in annotations if ann.indicator_value == indicator_value]
    
    return {
        "total_annotations": len(collaboration_platform.threat_annotations),
        "filtered_count": len(annotations),
        "annotations": [ann.dict() for ann in annotations]
    }

@app.get("/collaboration/users")
async def get_active_users():
    """Get active collaboration users"""
    return {
        "total_users": len(collaboration_platform.active_users),
        "users": [
            {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role,
                "status": user.status,
                "current_session": user.current_session,
                "last_activity": user.last_activity.isoformat()
            }
            for user in collaboration_platform.active_users.values()
        ]
    }

@app.get("/collaboration/analytics/overview")
async def get_collaboration_analytics():
    """Get collaboration platform analytics"""
    total_messages = sum(len(msgs) for msgs in collaboration_platform.message_history.values())
    active_sessions = len([s for s in collaboration_platform.collaboration_sessions.values() if s.status == SessionStatus.ACTIVE])
    
    # Recent activity (last hour)
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_messages = 0
    recent_annotations = 0
    
    for messages in collaboration_platform.message_history.values():
        recent_messages += len([
            msg for msg in messages 
            if datetime.fromisoformat(msg.timestamp) > one_hour_ago
        ])
    
    recent_annotations = len([
        ann for ann in collaboration_platform.threat_annotations
        if datetime.fromisoformat(ann.timestamp) > one_hour_ago
    ])
    
    return {
        "platform_stats": {
            "total_users": len(collaboration_platform.active_users),
            "total_sessions": len(collaboration_platform.collaboration_sessions),
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "total_annotations": len(collaboration_platform.threat_annotations),
            "total_incident_updates": len(collaboration_platform.incident_updates)
        },
        "recent_activity": {
            "messages_last_hour": recent_messages,
            "annotations_last_hour": recent_annotations,
            "active_connections": len(collaboration_platform.connection_manager.active_connections)
        },
        "session_analytics": {
            session_id: collaboration_platform.get_session_analytics(session_id)
            for session_id in collaboration_platform.collaboration_sessions.keys()
        }
    }

@app.get("/collaboration/demo", response_class=HTMLResponse)
async def collaboration_demo():
    """Real-time collaboration demo interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>XORB Real-time Collaboration Platform</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .collaboration-container { display: grid; grid-template-columns: 250px 1fr 300px; gap: 20px; height: 80vh; }
        .sidebar { background: #16213e; padding: 15px; border-radius: 8px; overflow-y: auto; }
        .main-chat { background: #0f3460; padding: 20px; border-radius: 8px; display: flex; flex-direction: column; }
        .activity-panel { background: #16213e; padding: 15px; border-radius: 8px; overflow-y: auto; }
        .chat-messages { flex: 1; overflow-y: auto; margin-bottom: 20px; background: #1a1a2e; padding: 15px; border-radius: 5px; }
        .message { margin: 10px 0; padding: 8px 12px; border-radius: 5px; background: #16213e; }
        .message.own { background: #0f3460; text-align: right; }
        .message-header { font-size: 0.8em; color: #888; margin-bottom: 5px; }
        .input-area { display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 10px; border: none; border-radius: 5px; background: #16213e; color: #eee; }
        .input-area button { padding: 10px 20px; border: none; border-radius: 5px; background: #e94560; color: white; cursor: pointer; }
        .user-list, .session-list { margin-bottom: 20px; }
        .user-item, .session-item { padding: 8px; margin: 5px 0; background: #0f3460; border-radius: 5px; cursor: pointer; }
        .user-item:hover, .session-item:hover { background: #1a1a2e; }
        .annotation-form { background: #16213e; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .annotation-form input, .annotation-form textarea { width: 100%; padding: 8px; margin: 5px 0; border: none; border-radius: 3px; background: #0f3460; color: #eee; }
        .threat-annotation { background: #2a0a0a; border-left: 4px solid #e94560; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
        .status-online { background: #4caf50; }
        .status-away { background: #ff9800; }
        .status-busy { background: #f44336; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤝 XORB REAL-TIME COLLABORATION PLATFORM</h1>
        <p>Multi-analyst cybersecurity operations workspace</p>
        <div id="connection-status">Connecting...</div>
    </div>
    
    <div class="collaboration-container">
        <div class="sidebar">
            <div class="user-list">
                <h3>👥 Active Users</h3>
                <div id="user-list"></div>
            </div>
            
            <div class="session-list">
                <h3>📋 Active Sessions</h3>
                <div id="session-list"></div>
            </div>
        </div>
        
        <div class="main-chat">
            <h3 id="session-title">💬 General Discussion</h3>
            <div class="chat-messages" id="chat-messages"></div>
            
            <div class="input-area">
                <input type="text" id="message-input" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="activity-panel">
            <h3>🛡️ Threat Annotations</h3>
            <div id="annotations-list"></div>
            
            <div class="annotation-form">
                <h4>Add Threat Annotation</h4>
                <input type="text" id="indicator-value" placeholder="Threat indicator">
                <select id="annotation-type">
                    <option value="malware">Malware</option>
                    <option value="phishing">Phishing</option>
                    <option value="c2">C2 Infrastructure</option>
                    <option value="apt">APT Activity</option>
                </select>
                <textarea id="annotation-content" placeholder="Annotation details"></textarea>
                <input type="range" id="confidence" min="0" max="1" step="0.1" value="0.8">
                <label>Confidence: <span id="confidence-value">0.8</span></label>
                <button onclick="addAnnotation()">Add Annotation</button>
            </div>
        </div>
    </div>
    
    <script>
        const userId = 'user_analyst_01'; // Default user
        const currentSessionId = 'session_demo_001';
        
        let ws = null;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://188.245.101.102:9005/ws/collaboration/${userId}?session_id=${currentSessionId}`);
            
            ws.onopen = function(event) {
                document.getElementById('connection-status').textContent = '✅ Connected to Collaboration Platform';
                document.getElementById('connection-status').style.color = '#4caf50';
                loadInitialData();
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.message_type === 'chat') {
                    addMessageToChat(data);
                } else if (data.type === 'threat_annotation') {
                    addAnnotationToList(data.data);
                } else if (data.type === 'connection_established') {
                    console.log('Connection established:', data);
                }
            };
            
            ws.onclose = function(event) {
                document.getElementById('connection-status').textContent = '❌ Disconnected';
                document.getElementById('connection-status').style.color = '#f44336';
                setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
            };
        }
        
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                const messageData = {
                    action: 'send_message',
                    session_id: currentSessionId,
                    content: message,
                    message_type: 'chat'
                };
                
                ws.send(JSON.stringify(messageData));
                
                // Add own message to chat immediately
                addMessageToChat({
                    username: 'You',
                    content: message,
                    timestamp: new Date().toISOString(),
                    message_type: 'chat'
                }, true);
                
                input.value = '';
            }
        }
        
        function addAnnotation() {
            const indicatorValue = document.getElementById('indicator-value').value;
            const annotationType = document.getElementById('annotation-type').value;
            const content = document.getElementById('annotation-content').value;
            const confidence = parseFloat(document.getElementById('confidence').value);
            
            if (indicatorValue && content && ws && ws.readyState === WebSocket.OPEN) {
                const annotationData = {
                    action: 'add_annotation',
                    session_id: currentSessionId,
                    indicator_value: indicatorValue,
                    annotation_type: annotationType,
                    content: content,
                    confidence: confidence
                };
                
                ws.send(JSON.stringify(annotationData));
                
                // Clear form
                document.getElementById('indicator-value').value = '';
                document.getElementById('annotation-content').value = '';
            }
        }
        
        function addMessageToChat(message, isOwn = false) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isOwn ? 'own' : ''}`;
            
            messageDiv.innerHTML = `
                <div class="message-header">${message.username} • ${new Date(message.timestamp).toLocaleTimeString()}</div>
                <div>${message.content}</div>
            `;
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function addAnnotationToList(annotation) {
            const annotationsList = document.getElementById('annotations-list');
            const annotationDiv = document.createElement('div');
            annotationDiv.className = 'threat-annotation';
            
            annotationDiv.innerHTML = `
                <div><strong>${annotation.annotation_type.toUpperCase()}</strong></div>
                <div>Indicator: ${annotation.indicator_value}</div>
                <div>Confidence: ${(annotation.confidence * 100).toFixed(0)}%</div>
                <div>${annotation.content}</div>
                <div style="font-size: 0.8em; color: #888;">${annotation.username} • ${new Date(annotation.timestamp).toLocaleTimeString()}</div>
            `;
            
            annotationsList.insertBefore(annotationDiv, annotationsList.firstChild);
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function updateConfidenceValue() {
            const confidence = document.getElementById('confidence').value;
            document.getElementById('confidence-value').textContent = confidence;
        }
        
        async function loadInitialData() {
            try {
                // Load users
                const usersResponse = await fetch('/collaboration/users');
                const usersData = await usersResponse.json();
                updateUserList(usersData.users);
                
                // Load sessions
                const sessionsResponse = await fetch('/collaboration/sessions');
                const sessionsData = await sessionsResponse.json();
                updateSessionList(sessionsData.sessions);
                
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }
        
        function updateUserList(users) {
            const userList = document.getElementById('user-list');
            userList.innerHTML = '';
            
            users.forEach(user => {
                const userDiv = document.createElement('div');
                userDiv.className = 'user-item';
                userDiv.innerHTML = `
                    <span class="status-indicator status-online"></span>
                    <strong>${user.username}</strong><br>
                    <small>${user.role}</small>
                `;
                userList.appendChild(userDiv);
            });
        }
        
        function updateSessionList(sessions) {
            const sessionList = document.getElementById('session-list');
            sessionList.innerHTML = '';
            
            sessions.forEach(session => {
                const sessionDiv = document.createElement('div');
                sessionDiv.className = 'session-item';
                sessionDiv.innerHTML = `
                    <strong>${session.session_name}</strong><br>
                    <small>${session.participants.length} participants</small>
                `;
                sessionList.appendChild(sessionDiv);
            });
        }
        
        // Update confidence slider display
        document.getElementById('confidence').addEventListener('input', updateConfidenceValue);
        
        // Initialize
        connectWebSocket();
        
        // Send heartbeat every 30 seconds
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: 'heartbeat'}));
            }
        }, 30000);
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """Collaboration platform health check"""
    return {
        "status": "healthy",
        "service": "xorb_collaboration_platform",
        "version": "4.0.0",
        "capabilities": [
            "Real-time Messaging",
            "Multi-user Sessions",
            "Threat Annotations",
            "Incident Collaboration",
            "WebSocket Streaming",
            "Role-based Permissions",
            "Session Analytics"
        ],
        "platform_stats": {
            "active_users": len(collaboration_platform.active_users),
            "active_sessions": len(collaboration_platform.collaboration_sessions),
            "websocket_connections": len(collaboration_platform.connection_manager.active_connections),
            "total_annotations": len(collaboration_platform.threat_annotations)
        }
    }

if __name__ == "__main__":
    # Create a demo session
    demo_session = collaboration_platform.create_collaboration_session(
        session_name="Demo Threat Analysis Session",
        session_type="threat_analysis",
        created_by="user_analyst_01",
        incident_id="incident_2025_001"
    )
    
    uvicorn.run(app, host="0.0.0.0", port=9005)