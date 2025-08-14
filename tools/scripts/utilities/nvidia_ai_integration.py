#!/usr/bin/env python3
"""
XORB NVIDIA AI Integration Service
Advanced AI capabilities using NVIDIA's Qwen3-235B model for cybersecurity operations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB NVIDIA AI Integration Service",
    description="Advanced AI capabilities using NVIDIA's Qwen3-235B model",
    version="9.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AITaskType(str, Enum):
    THREAT_ANALYSIS = "threat_analysis"
    INCIDENT_RESPONSE = "incident_response"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    SECURITY_PLANNING = "security_planning"
    COMPLIANCE_ANALYSIS = "compliance_analysis"
    MALWARE_ANALYSIS = "malware_analysis"
    FORENSICS_ANALYSIS = "forensics_analysis"
    RISK_ASSESSMENT = "risk_assessment"

class AIRequest(BaseModel):
    task_type: AITaskType
    query: str
    context: Dict = {}
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False

class AIResponse(BaseModel):
    task_id: str
    task_type: AITaskType
    query: str
    response: str
    reasoning: Optional[str] = None
    confidence: float
    processing_time: float
    timestamp: str
    metadata: Dict = {}

@dataclass
class AIConversation:
    conversation_id: str
    messages: List[Dict]
    created_at: datetime
    last_updated: datetime
    task_type: AITaskType

class NVIDIAAIService:
    """NVIDIA AI integration service for cybersecurity operations"""

    def __init__(self):
        # Initialize NVIDIA OpenAI client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )

        self.conversations: Dict[str, AIConversation] = {}
        self.ai_responses: List[AIResponse] = []

        # Task-specific system prompts
        self.system_prompts = {
            AITaskType.THREAT_ANALYSIS: """You are an expert cybersecurity threat analyst with deep knowledge of APTs, malware families, attack patterns, and threat intelligence. Analyze the provided threat data with forensic precision, identify indicators of compromise, assess threat severity, and provide actionable intelligence for defensive measures.""",

            AITaskType.INCIDENT_RESPONSE: """You are a senior incident response specialist with expertise in cyber incident containment, eradication, and recovery. Provide step-by-step incident response guidance, prioritize actions based on business impact, and recommend specific tools and techniques for effective incident handling.""",

            AITaskType.VULNERABILITY_ASSESSMENT: """You are a vulnerability assessment expert with comprehensive knowledge of CVEs, exploit techniques, and risk scoring methodologies. Analyze vulnerabilities with context of exploitability, business impact, and provide prioritized remediation recommendations with specific mitigation steps.""",

            AITaskType.SECURITY_PLANNING: """You are a strategic cybersecurity architect with expertise in security frameworks, compliance standards, and enterprise risk management. Develop comprehensive security strategies aligned with business objectives, industry best practices, and regulatory requirements.""",

            AITaskType.COMPLIANCE_ANALYSIS: """You are a compliance specialist with deep knowledge of security frameworks including SOC2, ISO27001, NIST, PCI-DSS, and HIPAA. Analyze compliance gaps, provide detailed remediation plans, and ensure alignment with regulatory requirements.""",

            AITaskType.MALWARE_ANALYSIS: """You are a malware reverse engineer with expertise in static and dynamic analysis, behavioral analysis, and threat attribution. Analyze malware samples, identify capabilities, extract IOCs, and provide comprehensive technical analysis reports.""",

            AITaskType.FORENSICS_ANALYSIS: """You are a digital forensics expert with specialized knowledge in incident investigation, evidence collection, and attack timeline reconstruction. Analyze forensic artifacts, reconstruct attack sequences, and provide legally sound investigative findings.""",

            AITaskType.RISK_ASSESSMENT: """You are a cybersecurity risk analyst with expertise in quantitative and qualitative risk assessment methodologies. Evaluate security risks, calculate business impact, and provide data-driven risk management recommendations."""
        }

    async def process_ai_request(self, request: AIRequest) -> AIResponse:
        """Process AI request using NVIDIA's Qwen3-235B model"""
        start_time = time.time()
        task_id = f"ai_task_{int(time.time())}_{len(self.ai_responses)}"

        # Get system prompt for task type
        system_prompt = self.system_prompts.get(request.task_type, "You are a cybersecurity expert assistant.")

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_user_query(request)}
        ]

        try:
            # Make request to NVIDIA API
            completion = self.client.chat.completions.create(
                model="qwen/qwen3-235b-a22b",
                messages=messages,
                temperature=request.temperature,
                top_p=1,
                max_tokens=request.max_tokens,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=False  # For now, we'll handle streaming separately
            )

            response_content = completion.choices[0].message.content
            reasoning_content = getattr(completion.choices[0].message, "reasoning_content", None)

            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(response_content, request.task_type)

            processing_time = time.time() - start_time

            ai_response = AIResponse(
                task_id=task_id,
                task_type=request.task_type,
                query=request.query,
                response=response_content,
                reasoning=reasoning_content,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "model": "qwen/qwen3-235b-a22b",
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "context_provided": bool(request.context)
                }
            )

            self.ai_responses.append(ai_response)
            return ai_response

        except Exception as e:
            # Handle API errors gracefully
            error_response = AIResponse(
                task_id=task_id,
                task_type=request.task_type,
                query=request.query,
                response=f"Error processing AI request: {str(e)}",
                reasoning=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                metadata={"error": str(e)}
            )

            self.ai_responses.append(error_response)
            return error_response

    async def stream_ai_response(self, request: AIRequest) -> AsyncGenerator[str, None]:
        """Stream AI response using NVIDIA's Qwen3-235B model"""
        system_prompt = self.system_prompts.get(request.task_type, "You are a cybersecurity expert assistant.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_user_query(request)}
        ]

        try:
            completion = self.client.chat.completions.create(
                model="qwen/qwen3-235b-a22b",
                messages=messages,
                temperature=request.temperature,
                top_p=1,
                max_tokens=request.max_tokens,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )

            full_response = ""
            full_reasoning = ""

            for chunk in completion:
                # Handle reasoning content
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    full_reasoning += reasoning
                    yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"

                # Handle response content
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'type': 'response', 'content': content})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'full_response': full_response, 'reasoning': full_reasoning})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    def _format_user_query(self, request: AIRequest) -> str:
        """Format user query with context"""
        formatted_query = request.query

        if request.context:
            context_str = "\n\nAdditional Context:\n"
            for key, value in request.context.items():
                context_str += f"- {key}: {value}\n"
            formatted_query += context_str

        return formatted_query

    def _calculate_confidence(self, response: str, task_type: AITaskType) -> float:
        """Calculate confidence score based on response characteristics"""
        confidence = 0.5  # Base confidence

        # Length-based confidence (longer responses often more detailed)
        if len(response) > 500:
            confidence += 0.1
        if len(response) > 1000:
            confidence += 0.1

        # Keyword-based confidence for different task types
        security_keywords = {
            AITaskType.THREAT_ANALYSIS: ["IOC", "indicator", "malware", "C2", "exfiltration", "persistence"],
            AITaskType.INCIDENT_RESPONSE: ["containment", "eradication", "recovery", "timeline", "evidence"],
            AITaskType.VULNERABILITY_ASSESSMENT: ["CVE", "CVSS", "exploit", "patch", "mitigation"],
            AITaskType.MALWARE_ANALYSIS: ["payload", "dropper", "behavior", "signature", "sandbox"],
            AITaskType.FORENSICS_ANALYSIS: ["artifact", "timeline", "evidence", "attribution", "chain of custody"]
        }

        task_keywords = security_keywords.get(task_type, [])
        keyword_matches = sum(1 for keyword in task_keywords if keyword.lower() in response.lower())
        confidence += min(0.3, keyword_matches * 0.05)

        # Structure-based confidence (presence of lists, numbers, specific recommendations)
        if any(marker in response for marker in ["1.", "2.", "3.", "•", "-", "*"]):
            confidence += 0.1

        return min(1.0, confidence)

    def create_conversation(self, task_type: AITaskType) -> str:
        """Create new AI conversation"""
        conversation_id = f"conv_{int(time.time())}_{len(self.conversations)}"

        conversation = AIConversation(
            conversation_id=conversation_id,
            messages=[],
            created_at=datetime.now(),
            last_updated=datetime.now(),
            task_type=task_type
        )

        self.conversations[conversation_id] = conversation
        return conversation_id

    async def continue_conversation(self, conversation_id: str, user_message: str) -> AIResponse:
        """Continue existing conversation"""
        if conversation_id not in self.conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation = self.conversations[conversation_id]
        conversation.messages.append({"role": "user", "content": user_message})
        conversation.last_updated = datetime.now()

        # Process with conversation context
        request = AIRequest(
            task_type=conversation.task_type,
            query=user_message,
            context={"conversation_history": conversation.messages[-10:]}  # Last 10 messages
        )

        response = await self.process_ai_request(request)
        conversation.messages.append({"role": "assistant", "content": response.response})

        return response

    def get_ai_analytics(self) -> Dict:
        """Get AI service analytics"""
        if not self.ai_responses:
            return {
                "total_requests": 0,
                "average_processing_time": 0,
                "average_confidence": 0,
                "task_type_distribution": {},
                "success_rate": 0
            }

        successful_responses = [r for r in self.ai_responses if "Error" not in r.response]

        task_distribution = {}
        for response in self.ai_responses:
            task_type = response.task_type.value
            task_distribution[task_type] = task_distribution.get(task_type, 0) + 1

        return {
            "total_requests": len(self.ai_responses),
            "successful_requests": len(successful_responses),
            "average_processing_time": round(sum(r.processing_time for r in self.ai_responses) / len(self.ai_responses), 3),
            "average_confidence": round(sum(r.confidence for r in self.ai_responses) / len(self.ai_responses), 3),
            "task_type_distribution": task_distribution,
            "success_rate": round(len(successful_responses) / len(self.ai_responses) * 100, 1),
            "active_conversations": len(self.conversations)
        }

# Initialize NVIDIA AI service
nvidia_ai_service = NVIDIAAIService()

@app.post("/ai/analyze")
async def analyze_with_ai(request: AIRequest):
    """Analyze cybersecurity data using NVIDIA AI"""
    response = await nvidia_ai_service.process_ai_request(request)
    return response.dict()

@app.post("/ai/stream")
async def stream_ai_analysis(request: AIRequest):
    """Stream AI analysis response"""
    if not request.stream:
        request.stream = True

    return StreamingResponse(
        nvidia_ai_service.stream_ai_response(request),
        media_type="text/plain"
    )

@app.post("/ai/conversation/create")
async def create_conversation(task_type: AITaskType):
    """Create new AI conversation"""
    conversation_id = nvidia_ai_service.create_conversation(task_type)
    return {
        "conversation_id": conversation_id,
        "task_type": task_type,
        "created_at": datetime.now().isoformat()
    }

@app.post("/ai/conversation/{conversation_id}/message")
async def send_message(conversation_id: str, message: str):
    """Send message to AI conversation"""
    response = await nvidia_ai_service.continue_conversation(conversation_id, message)
    return response.dict()

@app.get("/ai/conversations")
async def get_conversations():
    """Get all AI conversations"""
    conversations = []
    for conv in nvidia_ai_service.conversations.values():
        conversations.append({
            "conversation_id": conv.conversation_id,
            "task_type": conv.task_type.value,
            "message_count": len(conv.messages),
            "created_at": conv.created_at.isoformat(),
            "last_updated": conv.last_updated.isoformat()
        })

    return {
        "total_conversations": len(conversations),
        "conversations": conversations
    }

@app.get("/ai/responses")
async def get_ai_responses(limit: int = 20):
    """Get recent AI responses"""
    recent_responses = nvidia_ai_service.ai_responses[-limit:]
    return {
        "total_responses": len(nvidia_ai_service.ai_responses),
        "responses": [response.dict() for response in recent_responses]
    }

@app.get("/ai/analytics")
async def get_ai_analytics():
    """Get AI service analytics"""
    return nvidia_ai_service.get_ai_analytics()

@app.get("/ai/models")
async def get_available_models():
    """Get available AI models and capabilities"""
    return {
        "primary_model": "qwen/qwen3-235b-a22b",
        "model_info": {
            "name": "Qwen3-235B-A22B",
            "provider": "NVIDIA",
            "parameters": "235 billion",
            "context_length": "32,768 tokens",
            "capabilities": [
                "Advanced reasoning",
                "Multi-step problem solving",
                "Cybersecurity expertise",
                "Code analysis",
                "Technical documentation",
                "Threat intelligence analysis"
            ]
        },
        "supported_tasks": [task.value for task in AITaskType],
        "features": [
            "Streaming responses",
            "Reasoning traces",
            "Conversation memory",
            "Context-aware analysis",
            "Confidence scoring"
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def nvidia_ai_dashboard():
    """NVIDIA AI Integration Dashboard"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>XORB NVIDIA AI Integration</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0d1117; color: #f0f6fc; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .ai-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .card-title { font-size: 1.2em; font-weight: 600; color: #58a6ff; }
        .query-form { display: grid; gap: 15px; }
        .form-group { display: flex; flex-direction: column; gap: 5px; }
        .form-group label { color: #8b949e; font-size: 0.9em; }
        .form-group select, .form-group textarea, .form-group input { background: #0d1117; border: 1px solid #30363d; color: #f0f6fc; padding: 8px 12px; border-radius: 6px; }
        .form-group textarea { min-height: 100px; resize: vertical; }
        .analyze-btn { background: #238636; border: none; color: white; padding: 12px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; }
        .analyze-btn:hover { background: #2ea043; }
        .analyze-btn:disabled { background: #30363d; cursor: not-allowed; }
        .response-container { background: #0d1117; border-radius: 6px; padding: 15px; margin-top: 15px; max-height: 400px; overflow-y: auto; }
        .response-content { white-space: pre-wrap; line-height: 1.6; }
        .reasoning-content { background: #21262d; padding: 10px; border-radius: 4px; margin-bottom: 10px; border-left: 4px solid #d29922; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-top: 20px; }
        .metric { background: #0d1117; padding: 15px; border-radius: 6px; text-align: center; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #58a6ff; }
        .metric-label { font-size: 0.8em; color: #8b949e; margin-top: 5px; }
        .task-buttons { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
        .task-btn { background: #21262d; border: 1px solid #30363d; color: #f0f6fc; padding: 6px 12px; border-radius: 16px; cursor: pointer; font-size: 0.8em; }
        .task-btn.active { background: #58a6ff; color: #0d1117; }
        .loading { text-align: center; color: #8b949e; padding: 20px; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #30363d; border-radius: 50%; border-top-color: #58a6ff; animation: spin 1s ease-in-out infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .stream-toggle { display: flex; align-items: center; gap: 8px; }
        .stream-toggle input[type="checkbox"] { width: 16px; height: 16px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 XORB NVIDIA AI INTEGRATION</h1>
        <p>Advanced Cybersecurity AI using Qwen3-235B Model</p>
        <div id="status">Loading AI service...</div>
    </div>

    <div class="dashboard-grid">
        <!-- AI Analysis Card -->
        <div class="ai-card">
            <div class="card-header">
                <span class="card-title">🔍 AI Analysis</span>
            </div>
            <div class="query-form">
                <div class="form-group">
                    <label>Task Type</label>
                    <select id="task-type">
                        <option value="threat_analysis">Threat Analysis</option>
                        <option value="incident_response">Incident Response</option>
                        <option value="vulnerability_assessment">Vulnerability Assessment</option>
                        <option value="security_planning">Security Planning</option>
                        <option value="compliance_analysis">Compliance Analysis</option>
                        <option value="malware_analysis">Malware Analysis</option>
                        <option value="forensics_analysis">Forensics Analysis</option>
                        <option value="risk_assessment">Risk Assessment</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Query</label>
                    <textarea id="ai-query" placeholder="Enter your cybersecurity question or data to analyze..."></textarea>
                </div>
                <div class="form-group">
                    <label>Context (Optional)</label>
                    <textarea id="ai-context" placeholder="Additional context, data, or background information..."></textarea>
                </div>
                <div class="form-group">
                    <div class="stream-toggle">
                        <input type="checkbox" id="stream-mode">
                        <label for="stream-mode">Stream Response</label>
                    </div>
                </div>
                <button class="analyze-btn" onclick="analyzeWithAI()" id="analyze-btn">
                    Analyze with NVIDIA AI
                </button>
            </div>
            <div id="analysis-status" style="margin-top: 15px; color: #8b949e;"></div>
        </div>

        <!-- AI Analytics Card -->
        <div class="ai-card">
            <div class="card-header">
                <span class="card-title">📊 AI Analytics</span>
            </div>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="total-requests">-</div>
                    <div class="metric-label">Total Requests</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="success-rate">-</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="avg-confidence">-</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="avg-processing">-</div>
                    <div class="metric-label">Avg Time (s)</div>
                </div>
            </div>
            <div id="model-info" style="margin-top: 15px; font-size: 0.9em; color: #8b949e;">
                <strong>Model:</strong> Qwen3-235B-A22B via NVIDIA API<br>
                <strong>Capabilities:</strong> Advanced reasoning, cybersecurity expertise, streaming responses
            </div>
        </div>
    </div>

    <!-- Response Display -->
    <div class="ai-card">
        <div class="card-header">
            <span class="card-title">💬 AI Response</span>
            <button onclick="clearResponse()" style="background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 6px 12px; border-radius: 4px; cursor: pointer;">Clear</button>
        </div>
        <div class="response-container" id="response-container">
            <div class="loading">No analysis performed yet</div>
        </div>
    </div>

    <script>
        let currentAnalysis = null;
        let streamingResponse = false;

        async function loadDashboardData() {
            try {
                // Load AI analytics
                const analyticsResponse = await fetch('/ai/analytics');
                const analytics = await analyticsResponse.json();

                document.getElementById('total-requests').textContent = analytics.total_requests;
                document.getElementById('success-rate').textContent = analytics.success_rate + '%';
                document.getElementById('avg-confidence').textContent = (analytics.average_confidence * 100).toFixed(0) + '%';
                document.getElementById('avg-processing').textContent = analytics.average_processing_time;

                document.getElementById('status').textContent = '✅ NVIDIA AI Service Online';
                document.getElementById('status').style.color = '#2ea043';

            } catch (error) {
                console.error('Error loading dashboard data:', error);
                document.getElementById('status').textContent = '❌ Error Loading AI Service';
                document.getElementById('status').style.color = '#f85149';
            }
        }

        async function analyzeWithAI() {
            const taskType = document.getElementById('task-type').value;
            const query = document.getElementById('ai-query').value.trim();
            const context = document.getElementById('ai-context').value.trim();
            const streamMode = document.getElementById('stream-mode').checked;

            if (!query) {
                alert('Please enter a query to analyze');
                return;
            }

            const button = document.getElementById('analyze-btn');
            const status = document.getElementById('analysis-status');
            const responseContainer = document.getElementById('response-container');

            button.disabled = true;
            button.textContent = '🔄 Analyzing...';
            status.textContent = streamMode ? 'Streaming AI analysis...' : 'Processing AI analysis...';

            const requestData = {
                task_type: taskType,
                query: query,
                context: context ? { additional_info: context } : {},
                temperature: 0.7,
                max_tokens: 4096,
                stream: streamMode
            };

            try {
                if (streamMode) {
                    await streamAnalysis(requestData, responseContainer);
                } else {
                    await regularAnalysis(requestData, responseContainer);
                }

                status.textContent = '✅ Analysis complete';

                // Refresh analytics
                setTimeout(loadDashboardData, 1000);

            } catch (error) {
                status.textContent = '❌ Analysis failed: ' + error.message;
                responseContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            } finally {
                button.disabled = false;
                button.textContent = 'Analyze with NVIDIA AI';
            }
        }

        async function regularAnalysis(requestData, responseContainer) {
            const response = await fetch('/ai/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            let html = '';

            if (result.reasoning) {
                html += `<div class="reasoning-content"><strong>🤔 AI Reasoning:</strong><br>${result.reasoning}</div>`;
            }

            html += `<div class="response-content"><strong>🤖 AI Response:</strong><br>${result.response}</div>`;
            html += `<div style="margin-top: 10px; font-size: 0.8em; color: #8b949e;">`;
            html += `Confidence: ${(result.confidence * 100).toFixed(0)}% | `;
            html += `Processing Time: ${(result.processing_time * 1000).toFixed(0)}ms | `;
            html += `Task ID: ${result.task_id}`;
            html += `</div>`;

            responseContainer.innerHTML = html;
        }

        async function streamAnalysis(requestData, responseContainer) {
            const response = await fetch('/ai/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let reasoning = '';
            let content = '';

            responseContainer.innerHTML = '<div class="reasoning-content"><strong>🤔 AI Reasoning:</strong><br><span id="reasoning-text"></span></div><div class="response-content"><strong>🤖 AI Response:</strong><br><span id="response-text"></span></div>';

            const reasoningElement = document.getElementById('reasoning-text');
            const responseElement = document.getElementById('response-text');

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));

                                if (data.type === 'reasoning') {
                                    reasoning += data.content;
                                    reasoningElement.textContent = reasoning;
                                } else if (data.type === 'response') {
                                    content += data.content;
                                    responseElement.textContent = content;
                                } else if (data.type === 'complete') {
                                    console.log('Stream complete');
                                }
                            } catch (e) {
                                console.error('Error parsing streaming data:', e);
                            }
                        }
                    }

                    // Auto-scroll to bottom
                    responseContainer.scrollTop = responseContainer.scrollHeight;
                }
            } finally {
                reader.releaseLock();
            }
        }

        function clearResponse() {
            document.getElementById('response-container').innerHTML = '<div class="loading">No analysis performed yet</div>';
        }

        // Quick task selection
        function selectTask(taskType) {
            document.getElementById('task-type').value = taskType;

            // Update active button
            document.querySelectorAll('.task-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }

        // Sample queries for different task types
        const sampleQueries = {
            threat_analysis: "Analyze this suspicious network traffic: Multiple connections from IP 192.168.1.100 to external domains with DGA-like characteristics. Connection attempts to known C2 infrastructure detected.",
            incident_response: "We have detected a potential data breach. Unauthorized access to our database server was logged at 2:30 AM. What immediate steps should we take to contain and investigate this incident?",
            vulnerability_assessment: "Assess the risk of CVE-2024-1234 in our web application stack. We're running Apache 2.4.41 with PHP 8.1 and MySQL 8.0 on Ubuntu 20.04.",
            malware_analysis: "Analyze this suspicious file behavior: Process creates multiple registry entries, establishes network connections to pastebin.com, and drops additional executables in %TEMP% directory.",
            forensics_analysis: "Investigate timeline: User login at 3:15 AM, file access to sensitive documents at 3:17 AM, large data transfer at 3:22 AM, then immediate logout. Analyze this sequence."
        };

        // Auto-fill sample query when task type changes
        document.getElementById('task-type').addEventListener('change', function() {
            const taskType = this.value;
            const queryField = document.getElementById('ai-query');

            if (queryField.value.trim() === '' && sampleQueries[taskType]) {
                queryField.value = sampleQueries[taskType];
            }
        });

        // Initialize dashboard
        loadDashboardData();

        // Auto-refresh analytics every 30 seconds
        setInterval(loadDashboardData, 30000);

        // Set default sample query
        document.getElementById('ai-query').value = sampleQueries.threat_analysis;
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """NVIDIA AI integration health check"""
    try:
        # Test API connectivity
        test_completion = nvidia_ai_service.client.chat.completions.create(
            model="qwen/qwen3-235b-a22b",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )

        api_status = "healthy" if test_completion else "error"
    except Exception as e:
        api_status = f"error: {str(e)}"

    return {
        "status": "healthy" if api_status == "healthy" else "degraded",
        "service": "xorb_nvidia_ai_integration",
        "version": "9.0.0",
        "nvidia_api_status": api_status,
        "capabilities": [
            "Advanced AI Analysis",
            "Streaming Responses",
            "Reasoning Traces",
            "Multi-Task Support",
            "Conversation Memory",
            "Confidence Scoring",
            "Cybersecurity Expertise"
        ],
        "ai_stats": nvidia_ai_service.get_ai_analytics(),
        "model_info": {
            "name": "Qwen3-235B-A22B",
            "provider": "NVIDIA",
            "parameters": "235B",
            "context_length": "32K tokens"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9010)
