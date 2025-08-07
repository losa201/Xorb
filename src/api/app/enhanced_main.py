#!/usr/bin/env python3
"""
Enhanced XORB Neural Orchestrator API with Comprehensive Error Handling
Resilient FastAPI service with advanced error handling and recovery mechanisms
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Import error handling framework
sys.path.append('/root/Xorb')
from xorb_error_handling_framework import (
    XORBErrorHandler, ErrorCategory, ErrorSeverity, RecoveryStrategy,
    RecoveryAction, xorb_async_error_handler, get_error_handler
)

# Prometheus instrumentation with error handling
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
    
    # Custom metrics
    ERROR_COUNTER = Counter('xorb_api_errors_total', 'Total API errors', ['service', 'category', 'severity'])
    RESPONSE_TIME = Histogram('xorb_api_response_time_seconds', 'API response time')
    ACTIVE_REQUESTS = Gauge('xorb_api_active_requests', 'Active API requests')
    CIRCUIT_BREAKER_STATE = Gauge('xorb_circuit_breaker_state', 'Circuit breaker state', ['service'])
    
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: Prometheus instrumentation not available")

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling"""
    
    def __init__(self, app, error_handler: XORBErrorHandler):
        super().__init__(app)
        self.error_handler = error_handler
        
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request context
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        if PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.inc()
        
        try:
            response = await call_next(request)
            
            # Log successful requests
            duration = time.time() - start_time
            
            if PROMETHEUS_AVAILABLE:
                RESPONSE_TIME.observe(duration)
            
            # Add error handling headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Handle unexpected errors
            error_context = self.error_handler.handle_error(
                e,
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.HIGH,
                context={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "user_agent": request.headers.get("user-agent"),
                    "duration": time.time() - start_time
                }
            )
            
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(
                    service="neural_orchestrator",
                    category=error_context.category.value,
                    severity=error_context.severity.value
                ).inc()
            
            # Return user-friendly error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_id": error_context.error_id,
                    "request_id": request_id,
                    "timestamp": error_context.timestamp.isoformat(),
                    "message": "An unexpected error occurred. Please try again later."
                },
                headers={"X-Request-ID": request_id}
            )
        finally:
            if PROMETHEUS_AVAILABLE:
                ACTIVE_REQUESTS.dec()

class HealthChecker:
    """Health checking with circuit breaker patterns"""
    
    def __init__(self, error_handler: XORBErrorHandler):
        self.error_handler = error_handler
        self.dependencies = {
            "database": self._check_database,
            "redis": self._check_redis,
            "neural_models": self._check_neural_models,
            "external_apis": self._check_external_apis
        }
        self.health_status = {}
        
        # Register recovery actions
        self._register_recovery_actions()
    
    def _register_recovery_actions(self):
        """Register health check recovery actions"""
        
        # Database connection recovery
        db_recovery = RecoveryAction(
            action_id="db_reconnect",
            name="Database Reconnection",
            strategy=RecoveryStrategy.RETRY,
            handler=self._recover_database_connection,
            max_attempts=3,
            conditions={"categories": ["database"]}
        )
        self.error_handler.register_recovery_action(db_recovery)
        
        # Redis connection recovery
        redis_recovery = RecoveryAction(
            action_id="redis_reconnect",
            name="Redis Reconnection",
            strategy=RecoveryStrategy.RETRY,
            handler=self._recover_redis_connection,
            max_attempts=3,
            conditions={"categories": ["external_service"]}
        )
        self.error_handler.register_recovery_action(redis_recovery)
        
        # Neural model fallback
        model_fallback = RecoveryAction(
            action_id="model_fallback",
            name="Neural Model Fallback",
            strategy=RecoveryStrategy.FALLBACK,
            handler=self._fallback_neural_model,
            conditions={"categories": ["business_logic"]}
        )
        self.error_handler.register_recovery_action(model_fallback)
    
    async def _recover_database_connection(self, error_context) -> bool:
        """Attempt to recover database connection"""
        try:
            # Simulate database reconnection
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False
    
    async def _recover_redis_connection(self, error_context) -> bool:
        """Attempt to recover Redis connection"""
        try:
            # Simulate Redis reconnection
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False
    
    async def _fallback_neural_model(self, error_context) -> bool:
        """Fallback to basic neural model"""
        try:
            # Implement fallback logic
            return True
        except Exception:
            return False
    
    @xorb_async_error_handler(
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.HIGH,
        retry_count=2
    )
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Simulate database check
            await asyncio.sleep(0.01)
            return {"status": "healthy", "response_time": 0.01}
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")
    
    @xorb_async_error_handler(
        category=ErrorCategory.EXTERNAL_SERVICE,
        severity=ErrorSeverity.MEDIUM,
        retry_count=1
    )
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            # Simulate Redis check
            await asyncio.sleep(0.005)
            return {"status": "healthy", "response_time": 0.005}
        except Exception as e:
            raise ConnectionError(f"Redis connection failed: {e}")
    
    @xorb_async_error_handler(
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.MEDIUM,
        retry_count=1
    )
    async def _check_neural_models(self) -> Dict[str, Any]:
        """Check neural models status"""
        try:
            # Simulate model check
            return {
                "status": "healthy",
                "models_loaded": 3,
                "memory_usage": "2.1GB"
            }
        except Exception as e:
            raise RuntimeError(f"Neural model check failed: {e}")
    
    @xorb_async_error_handler(
        category=ErrorCategory.EXTERNAL_SERVICE,
        severity=ErrorSeverity.LOW,
        retry_count=1
    )
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API dependencies"""
        try:
            # Simulate external API check
            return {"status": "healthy", "external_services": 2}
        except Exception as e:
            raise ConnectionError(f"External API check failed: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        overall_status = "healthy"
        checks = {}
        
        for name, check_func in self.dependencies.items():
            try:
                result = await check_func()
                checks[name] = result
            except Exception as e:
                overall_status = "degraded"
                checks[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "degraded": self.error_handler.is_service_degraded(check_func.__name__)
                }
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "service": "neural_orchestrator",
            "version": "2.0.0",
            "checks": checks,
            "error_summary": self.error_handler.get_error_summary()
        }

class NeuralOrchestrator:
    """Enhanced Neural Orchestrator with error handling"""
    
    def __init__(self, error_handler: XORBErrorHandler):
        self.error_handler = error_handler
        self.active_agents = 0
        self.neural_models_loaded = 0
        self.performance_metrics = {
            "requests_processed": 0,
            "errors_handled": 0,
            "average_response_time": 0.0
        }
        
        # Register recovery actions for orchestration
        self._register_orchestration_recovery_actions()
    
    def _register_orchestration_recovery_actions(self):
        """Register orchestration-specific recovery actions"""
        
        # Agent scaling recovery
        agent_scaling = RecoveryAction(
            action_id="scale_agents",
            name="Scale Neural Agents",
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            handler=self._scale_agents_recovery,
            conditions={"categories": ["system_resource"]}
        )
        self.error_handler.register_recovery_action(agent_scaling)
        
        # Model reload recovery
        model_reload = RecoveryAction(
            action_id="reload_models",
            name="Reload Neural Models",
            strategy=RecoveryStrategy.RETRY,
            handler=self._reload_models_recovery,
            max_attempts=2,
            conditions={"categories": ["business_logic"]}
        )
        self.error_handler.register_recovery_action(model_reload)
    
    async def _scale_agents_recovery(self, error_context) -> bool:
        """Recovery action to scale neural agents"""
        try:
            if self.active_agents > 10:
                self.active_agents = max(5, self.active_agents // 2)
                return True
            return False
        except Exception:
            return False
    
    async def _reload_models_recovery(self, error_context) -> bool:
        """Recovery action to reload neural models"""
        try:
            # Simulate model reload
            self.neural_models_loaded = 3
            return True
        except Exception:
            return False
    
    @xorb_async_error_handler(
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.HIGH,
        retry_count=2
    )
    async def orchestrate_agents(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main orchestration function with error handling"""
        try:
            # Simulate agent orchestration
            task_id = str(uuid.uuid4())
            
            # Check if service is degraded
            if self.error_handler.is_service_degraded("orchestrate_agents"):
                return {
                    "task_id": task_id,
                    "status": "degraded",
                    "message": "Service running in degraded mode",
                    "agents_allocated": min(2, self.active_agents),
                    "estimated_completion": "extended"
                }
            
            # Normal orchestration
            agents_needed = request_data.get("complexity", 3)
            
            if agents_needed > 10:
                raise ValueError("Too many agents requested")
            
            # Simulate orchestration work
            await asyncio.sleep(0.1)
            
            self.active_agents += agents_needed
            self.performance_metrics["requests_processed"] += 1
            
            return {
                "task_id": task_id,
                "status": "success",
                "agents_allocated": agents_needed,
                "estimated_completion": "2-5 minutes",
                "neural_models_used": self.neural_models_loaded
            }
            
        except ValueError as e:
            # Handle validation errors
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Handle unexpected errors
            self.performance_metrics["errors_handled"] += 1
            raise RuntimeError(f"Orchestration failed: {e}")
    
    @xorb_async_error_handler(
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.MEDIUM,
        retry_count=1
    )
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "active_agents": self.active_agents,
            "neural_models_loaded": self.neural_models_loaded,
            "performance_metrics": self.performance_metrics,
            "service_health": "healthy" if not self.error_handler.is_service_degraded("orchestrate_agents") else "degraded"
        }

# Global error handler
error_handler = get_error_handler("neural_orchestrator")
health_checker = HealthChecker(error_handler)
orchestrator = NeuralOrchestrator(error_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("🚀 Starting Enhanced Neural Orchestrator...")
    
    # Initialize Prometheus instrumentation
    if PROMETHEUS_AVAILABLE:
        instrumentator = Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=[".*admin.*", "/metrics"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="xorb_api_requests_inprogress",
            inprogress_labels=True,
        )
        instrumentator.instrument(app).expose(app)
        print("📊 Prometheus instrumentation enabled")
    
    print("✅ Neural Orchestrator startup complete")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Enhanced Neural Orchestrator...")
    
    # Generate final error report
    error_summary = error_handler.get_error_summary()
    print(f"📋 Final Error Summary: {json.dumps(error_summary, indent=2, default=str)}")
    
    print("✅ Neural Orchestrator shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Enhanced XORB Neural Orchestrator",
    description="Resilient Neural Orchestration API with comprehensive error handling",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware, error_handler=error_handler)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handlers
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error tracking"""
    error_context = error_handler.handle_error(
        Exception(f"HTTP {exc.status_code}: {exc.detail}"),
        category=ErrorCategory.VALIDATION if exc.status_code < 500 else ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.LOW if exc.status_code < 500 else ErrorSeverity.HIGH,
        context={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )
    
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNTER.labels(
            service="neural_orchestrator",
            category=error_context.category.value,
            severity=error_context.severity.value
        ).inc()
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_id": error_context.error_id,
            "status_code": exc.status_code,
            "timestamp": error_context.timestamp.isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    error_context = error_handler.handle_error(
        exc,
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.LOW,
        context={
            "validation_errors": exc.errors(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )
    
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNTER.labels(
            service="neural_orchestrator",
            category="validation",
            severity="low"
        ).inc()
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "error_id": error_context.error_id,
            "details": exc.errors(),
            "timestamp": error_context.timestamp.isoformat()
        }
    )

# API Routes
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return await health_checker.get_health_status()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enhanced XORB Neural Orchestrator",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Advanced Error Handling",
            "Circuit Breaker Patterns",
            "Graceful Degradation",
            "Comprehensive Monitoring",
            "Automatic Recovery"
        ]
    }

@app.post("/orchestrate")
async def orchestrate_task(request_data: Dict[str, Any]):
    """Main orchestration endpoint"""
    return await orchestrator.orchestrate_agents(request_data)

@app.get("/agents/status")
async def get_agents_status():
    """Get current agent status"""
    return await orchestrator.get_agent_status()

@app.get("/errors/summary")
async def get_error_summary():
    """Get error handling summary"""
    return error_handler.get_error_summary()

@app.post("/errors/clear-degradation/{function_name}")
async def clear_degradation(function_name: str):
    """Clear degradation for a specific function"""
    error_handler.clear_degradation(function_name)
    return {"message": f"Degradation cleared for {function_name}"}

@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get performance metrics"""
    return {
        "service": "neural_orchestrator",
        "metrics": orchestrator.performance_metrics,
        "circuit_breakers": {
            service: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count
            }
            for service, cb in error_handler.circuit_breakers.items()
        },
        "error_summary": error_handler.get_error_summary()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🧠 Starting Enhanced XORB Neural Orchestrator...")
    uvicorn.run(
        "enhanced_main:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info",
        access_log=True
    )