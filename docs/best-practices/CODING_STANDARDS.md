# XORB Platform Coding Standards

## Overview

This document defines the coding standards and best practices for the XORB platform. Following these standards ensures consistency, maintainability, and quality across the codebase.

---

## General Principles

### SOLID Principles

#### 1. Single Responsibility Principle (SRP)
- Each class should have only one reason to change
- Separate concerns into different classes
- Example:
```python
# Good - Single responsibility
class UserRepository:
    def save(self, user: User) -> User:
        # Only handles user persistence
        pass

class UserValidator:
    def validate(self, user: User) -> bool:
        # Only handles user validation
        pass

# Bad - Multiple responsibilities
class UserManager:
    def save(self, user: User) -> User:
        # Handles both validation AND persistence
        if self.validate(user):
            # Save logic
            pass

    def validate(self, user: User) -> bool:
        # Validation logic
        pass
```

#### 2. Open/Closed Principle (OCP)
- Open for extension, closed for modification
- Use interfaces and abstract classes
- Example:
```python
from abc import ABC, abstractmethod

class ThreatDetector(ABC):
    @abstractmethod
    def detect(self, data: Any) -> ThreatResult:
        pass

class MalwareDetector(ThreatDetector):
    def detect(self, data: Any) -> ThreatResult:
        # Malware detection logic
        pass

class IntrusionDetector(ThreatDetector):
    def detect(self, data: Any) -> ThreatResult:
        # Intrusion detection logic
        pass
```

#### 3. Liskov Substitution Principle (LSP)
- Derived classes must be substitutable for base classes
- Maintain behavioral contracts
- Example:
```python
class SecurityEvent:
    def process(self) -> bool:
        return True

class CriticalSecurityEvent(SecurityEvent):
    def process(self) -> bool:
        # Must return bool, not break the contract
        self.escalate()
        return super().process()
```

#### 4. Interface Segregation Principle (ISP)
- Many specific interfaces are better than one general interface
- Example:
```python
# Good - Segregated interfaces
class Readable(Protocol):
    def read(self) -> str: ...

class Writable(Protocol):
    def write(self, data: str) -> None: ...

# Bad - Fat interface
class FileHandler(Protocol):
    def read(self) -> str: ...
    def write(self, data: str) -> None: ...
    def compress(self) -> None: ...
    def encrypt(self) -> None: ...
```

#### 5. Dependency Inversion Principle (DIP)
- Depend on abstractions, not concretions
- Use dependency injection
- Example:
```python
# Good - Depends on abstraction
class ThreatAnalysisService:
    def __init__(self, detector: ThreatDetector):
        self.detector = detector

    def analyze(self, data: Any) -> ThreatResult:
        return self.detector.detect(data)

# Bad - Depends on concrete implementation
class ThreatAnalysisService:
    def __init__(self):
        self.detector = MalwareDetector()  # Tight coupling
```

---

## Python Standards

### Code Formatting
- **Line Length**: 100 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized with isort
- **Formatting**: Black code formatter

```python
# Good formatting example
from typing import Dict, List, Optional
from uuid import UUID

from src.domain.entities.base import AggregateRoot
from src.domain.value_objects.security import ThreatLevel


class SecurityIncident(AggregateRoot):
    """Represents a security incident in the system."""

    def __init__(
        self,
        incident_id: UUID,
        threat_level: ThreatLevel,
        description: str,
        affected_assets: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(incident_id)
        self.threat_level = threat_level
        self.description = description
        self.affected_assets = affected_assets
        self.metadata = metadata or {}
```

### Naming Conventions
- **Classes**: PascalCase (`SecurityIncident`)
- **Functions/Methods**: snake_case (`analyze_threat`)
- **Variables**: snake_case (`threat_level`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_RETRY_ATTEMPTS`)
- **Private Members**: Leading underscore (`_internal_method`)

### Type Hints
- All functions must have type hints
- Use `typing` module for complex types
- Example:
```python
from typing import Dict, List, Optional, Union
from uuid import UUID

async def process_security_events(
    events: List[SecurityEvent],
    filters: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0
) -> Dict[str, Union[int, List[str]]]:
    """Process a list of security events with optional filtering."""
    # Implementation
    pass
```

### Documentation
- **Docstrings**: All public classes and functions
- **Format**: Google style docstrings
- **Examples**:

```python
class ThreatIntelligenceEngine:
    """Engine for processing threat intelligence data.

    This class provides methods for analyzing, correlating, and enriching
    threat intelligence data from multiple sources.

    Attributes:
        sources: List of configured threat intelligence sources
        correlation_rules: Rules for correlating threat indicators
    """

    def analyze_indicators(
        self,
        indicators: List[ThreatIndicator],
        context: Optional[AnalysisContext] = None
    ) -> ThreatAnalysisResult:
        """Analyze threat indicators for potential threats.

        Args:
            indicators: List of threat indicators to analyze
            context: Optional context for analysis (geolocation, timeframe, etc.)

        Returns:
            ThreatAnalysisResult containing analysis results and confidence scores

        Raises:
            AnalysisException: If analysis fails due to invalid indicators
            TimeoutException: If analysis exceeds timeout threshold

        Example:
            >>> engine = ThreatIntelligenceEngine()
            >>> indicators = [ThreatIndicator(type="ip", value="192.168.1.1")]
            >>> result = engine.analyze_indicators(indicators)
            >>> print(result.threat_score)
            0.85
        """
        # Implementation
        pass
```

### Error Handling
- Use custom exceptions for domain-specific errors
- Fail fast principle
- Comprehensive error context
- Example:

```python
class ThreatAnalysisException(Exception):
    """Exception raised during threat analysis."""

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

async def analyze_threat(indicator: ThreatIndicator) -> ThreatResult:
    """Analyze a threat indicator."""
    try:
        if not indicator.is_valid():
            raise ThreatAnalysisException(
                message="Invalid threat indicator",
                error_code="INVALID_INDICATOR",
                details={"indicator": indicator.to_dict()}
            )

        # Analysis logic
        result = await _perform_analysis(indicator)
        return result

    except ValidationError as e:
        raise ThreatAnalysisException(
            message=f"Validation failed: {str(e)}",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": e.errors()}
        ) from e
    except Exception as e:
        # Log the unexpected error
        logger.exception("Unexpected error during threat analysis")
        raise ThreatAnalysisException(
            message="Internal analysis error",
            error_code="INTERNAL_ERROR",
            details={"original_error": str(e)}
        ) from e
```

---

## Architecture Patterns

### Repository Pattern
```python
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

class SecurityEventRepository(ABC):
    """Abstract repository for security events."""

    @abstractmethod
    async def save(self, event: SecurityEvent) -> SecurityEvent:
        """Save a security event."""
        pass

    @abstractmethod
    async def get_by_id(self, event_id: UUID) -> Optional[SecurityEvent]:
        """Get security event by ID."""
        pass

    @abstractmethod
    async def find_by_criteria(
        self,
        criteria: SecurityEventCriteria
    ) -> List[SecurityEvent]:
        """Find security events by criteria."""
        pass

class PostgreSQLSecurityEventRepository(SecurityEventRepository):
    """PostgreSQL implementation of security event repository."""

    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool

    async def save(self, event: SecurityEvent) -> SecurityEvent:
        """Save security event to PostgreSQL."""
        # Implementation
        pass
```

### Use Case Pattern
```python
from dataclasses import dataclass
from typing import List

@dataclass
class AnalyzeThreatRequest:
    """Request for threat analysis use case."""
    indicators: List[ThreatIndicator]
    priority: int = 1
    requester_id: UUID = None

@dataclass
class AnalyzeThreatResponse:
    """Response from threat analysis use case."""
    analysis_id: UUID
    threat_score: float
    recommendations: List[str]
    confidence: float

class AnalyzeThreatUseCase:
    """Use case for analyzing potential threats."""

    def __init__(
        self,
        threat_repository: ThreatRepository,
        analysis_engine: ThreatAnalysisEngine,
        notification_service: NotificationService
    ):
        self.threat_repository = threat_repository
        self.analysis_engine = analysis_engine
        self.notification_service = notification_service

    async def execute(
        self,
        request: AnalyzeThreatRequest
    ) -> AnalyzeThreatResponse:
        """Execute threat analysis use case."""
        # Validate input
        if not request.indicators:
            raise ValidationException("No indicators provided")

        # Perform analysis
        analysis_result = await self.analysis_engine.analyze(
            request.indicators
        )

        # Save results
        threat_record = await self.threat_repository.save(analysis_result)

        # Send notifications if high threat
        if analysis_result.threat_score > 0.8:
            await self.notification_service.send_alert(threat_record)

        return AnalyzeThreatResponse(
            analysis_id=threat_record.id,
            threat_score=analysis_result.threat_score,
            recommendations=analysis_result.recommendations,
            confidence=analysis_result.confidence
        )
```

### Command Query Responsibility Segregation (CQRS)
```python
from abc import ABC, abstractmethod

# Command side (Write operations)
class Command(ABC):
    """Base command class."""
    pass

class CreateSecurityIncidentCommand(Command):
    """Command to create a security incident."""

    def __init__(
        self,
        incident_type: str,
        severity: str,
        description: str,
        affected_systems: List[str]
    ):
        self.incident_type = incident_type
        self.severity = severity
        self.description = description
        self.affected_systems = affected_systems

class CommandHandler(ABC):
    """Base command handler."""

    @abstractmethod
    async def handle(self, command: Command) -> Any:
        pass

class CreateSecurityIncidentHandler(CommandHandler):
    """Handler for creating security incidents."""

    async def handle(
        self,
        command: CreateSecurityIncidentCommand
    ) -> SecurityIncident:
        # Create and save incident
        incident = SecurityIncident.create(
            incident_type=command.incident_type,
            severity=command.severity,
            description=command.description,
            affected_systems=command.affected_systems
        )

        return await self.incident_repository.save(incident)

# Query side (Read operations)
class Query(ABC):
    """Base query class."""
    pass

class GetSecurityIncidentsQuery(Query):
    """Query to get security incidents."""

    def __init__(
        self,
        severity: Optional[str] = None,
        date_range: Optional[DateRange] = None,
        limit: int = 100
    ):
        self.severity = severity
        self.date_range = date_range
        self.limit = limit

class QueryHandler(ABC):
    """Base query handler."""

    @abstractmethod
    async def handle(self, query: Query) -> Any:
        pass

class GetSecurityIncidentsHandler(QueryHandler):
    """Handler for getting security incidents."""

    async def handle(
        self,
        query: GetSecurityIncidentsQuery
    ) -> List[SecurityIncidentView]:
        # Query optimized read model
        return await self.incident_query_service.get_incidents(
            severity=query.severity,
            date_range=query.date_range,
            limit=query.limit
        )
```

---

## Testing Standards

### Test Structure
- Follow AAA pattern: Arrange, Act, Assert
- One assertion per test
- Descriptive test names
- Example:

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestThreatAnalysisService:
    """Test suite for ThreatAnalysisService."""

    @pytest.fixture
    def mock_threat_repository(self):
        """Mock threat repository."""
        return AsyncMock(spec=ThreatRepository)

    @pytest.fixture
    def mock_analysis_engine(self):
        """Mock analysis engine."""
        return AsyncMock(spec=ThreatAnalysisEngine)

    @pytest.fixture
    def service(self, mock_threat_repository, mock_analysis_engine):
        """Create service instance with mocked dependencies."""
        return ThreatAnalysisService(
            threat_repository=mock_threat_repository,
            analysis_engine=mock_analysis_engine
        )

    async def test_analyze_threat_with_valid_indicator_returns_analysis_result(
        self,
        service,
        mock_analysis_engine
    ):
        """Test that analyzing a valid threat indicator returns analysis result."""
        # Arrange
        indicator = ThreatIndicator(type="ip", value="192.168.1.1")
        expected_result = ThreatAnalysisResult(
            threat_score=0.85,
            confidence=0.95
        )
        mock_analysis_engine.analyze.return_value = expected_result

        # Act
        result = await service.analyze_threat(indicator)

        # Assert
        assert result.threat_score == 0.85
        assert result.confidence == 0.95
        mock_analysis_engine.analyze.assert_called_once_with(indicator)

    async def test_analyze_threat_with_invalid_indicator_raises_validation_exception(
        self,
        service
    ):
        """Test that analyzing an invalid indicator raises ValidationException."""
        # Arrange
        invalid_indicator = ThreatIndicator(type="", value="")

        # Act & Assert
        with pytest.raises(ValidationException) as exc_info:
            await service.analyze_threat(invalid_indicator)

        assert "Invalid threat indicator" in str(exc_info.value)
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Security Tests**: Test security controls
- **Performance Tests**: Test performance characteristics

### Test Data
- Use factories for test data creation
- Avoid hardcoded values
- Example:

```python
from factory import Factory, Faker, SubFactory
from factory.alchemy import SQLAlchemyModelFactory

class ThreatIndicatorFactory(Factory):
    """Factory for creating threat indicators."""

    class Meta:
        model = ThreatIndicator

    type = Faker('random_element', elements=['ip', 'domain', 'hash'])
    value = Faker('ipv4')
    confidence = Faker('pyfloat', min_value=0.0, max_value=1.0)
    source = Faker('company')

class SecurityIncidentFactory(SQLAlchemyModelFactory):
    """Factory for creating security incidents."""

    class Meta:
        model = SecurityIncident
        sqlalchemy_session_persistence = 'commit'

    incident_type = Faker('random_element', elements=['malware', 'intrusion', 'ddos'])
    severity = Faker('random_element', elements=['low', 'medium', 'high', 'critical'])
    description = Faker('text', max_nb_chars=500)
    status = 'open'

    # Relationships
    primary_indicator = SubFactory(ThreatIndicatorFactory)
```

---

## Performance Standards

### Async/Await Best Practices
```python
import asyncio
from typing import List

# Good - Concurrent execution
async def process_multiple_threats(
    indicators: List[ThreatIndicator]
) -> List[ThreatResult]:
    """Process multiple threats concurrently."""
    tasks = [
        analyze_single_threat(indicator)
        for indicator in indicators
    ]
    return await asyncio.gather(*tasks)

# Bad - Sequential execution
async def process_multiple_threats_sequential(
    indicators: List[ThreatIndicator]
) -> List[ThreatResult]:
    """Process threats sequentially (slower)."""
    results = []
    for indicator in indicators:
        result = await analyze_single_threat(indicator)
        results.append(result)
    return results
```

### Database Optimization
```python
# Good - Batch operations
async def save_multiple_events(
    events: List[SecurityEvent]
) -> List[SecurityEvent]:
    """Save multiple events in a single transaction."""
    async with self.get_transaction() as transaction:
        return await transaction.executemany(
            "INSERT INTO security_events (...) VALUES (...)",
            [event.to_dict() for event in events]
        )

# Good - Use connection pooling
class DatabaseRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def execute_query(self, query: str, *params):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *params)
```

### Caching Strategies
```python
from functools import lru_cache
import redis.asyncio as redis

class ThreatIntelligenceService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    @lru_cache(maxsize=1000)
    def get_static_threat_data(self, threat_type: str) -> dict:
        """Cache static threat data in memory."""
        return self._load_static_data(threat_type)

    async def get_threat_reputation(self, indicator: str) -> float:
        """Cache threat reputation with TTL."""
        cache_key = f"reputation:{indicator}"

        # Check cache first
        cached_score = await self.redis.get(cache_key)
        if cached_score:
            return float(cached_score)

        # Calculate and cache
        score = await self._calculate_reputation(indicator)
        await self.redis.setex(cache_key, 3600, score)  # 1 hour TTL
        return score
```

---

## Security Standards

### Input Validation
```python
from pydantic import BaseModel, validator

class ThreatIndicatorInput(BaseModel):
    """Input validation for threat indicators."""

    type: str
    value: str
    confidence: float

    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['ip', 'domain', 'hash', 'url']
        if v not in allowed_types:
            raise ValueError(f'Type must be one of: {allowed_types}')
        return v

    @validator('value')
    def validate_value(cls, v, values):
        if values.get('type') == 'ip':
            import ipaddress
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError('Invalid IP address format')
        return v

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
```

### Secret Management
```python
import os
from src.infrastructure.security.vault_client import VaultClient

class SecurityConfig:
    """Secure configuration management."""

    def __init__(self, vault_client: VaultClient):
        self.vault = vault_client

    async def get_secret(self, secret_path: str) -> str:
        """Get secret from vault with fallback to environment."""
        try:
            return await self.vault.get_secret(secret_path)
        except Exception:
            # Fallback to environment variable
            env_var = secret_path.replace('/', '_').upper()
            secret = os.getenv(env_var)
            if not secret:
                raise ValueError(f"Secret not found: {secret_path}")
            return secret

    def get_database_url(self) -> str:
        """Get database URL without exposing password in logs."""
        return os.getenv('DATABASE_URL', '').replace(
            '://', '://***:***@', 1
        ) if 'DATABASE_URL' in os.environ else 'Not configured'
```

---

## Monitoring and Logging

### Structured Logging
```python
import structlog
from typing import Any, Dict

logger = structlog.get_logger()

class ThreatAnalysisService:
    async def analyze_threat(self, indicator: ThreatIndicator) -> ThreatResult:
        """Analyze threat with structured logging."""

        logger.info(
            "Starting threat analysis",
            indicator_type=indicator.type,
            indicator_value_hash=self._hash_sensitive_value(indicator.value),
            request_id=self.request_id
        )

        try:
            result = await self._perform_analysis(indicator)

            logger.info(
                "Threat analysis completed",
                threat_score=result.threat_score,
                confidence=result.confidence,
                analysis_duration_ms=result.duration_ms,
                request_id=self.request_id
            )

            return result

        except Exception as e:
            logger.error(
                "Threat analysis failed",
                error_type=type(e).__name__,
                error_message=str(e),
                indicator_type=indicator.type,
                request_id=self.request_id,
                exc_info=True
            )
            raise
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
threat_analysis_total = Counter(
    'threat_analysis_total',
    'Total number of threat analyses',
    ['indicator_type', 'result']
)

threat_analysis_duration = Histogram(
    'threat_analysis_duration_seconds',
    'Time spent analyzing threats',
    ['indicator_type']
)

active_threats = Gauge(
    'active_threats_total',
    'Number of active threats'
)

class ThreatAnalysisService:
    async def analyze_threat(self, indicator: ThreatIndicator) -> ThreatResult:
        """Analyze threat with metrics collection."""

        with threat_analysis_duration.labels(
            indicator_type=indicator.type
        ).time():
            try:
                result = await self._perform_analysis(indicator)

                threat_analysis_total.labels(
                    indicator_type=indicator.type,
                    result='success'
                ).inc()

                if result.threat_score > 0.8:
                    active_threats.inc()

                return result

            except Exception as e:
                threat_analysis_total.labels(
                    indicator_type=indicator.type,
                    result='error'
                ).inc()
                raise
```

---

## Code Review Checklist

### Functionality
- [ ] Code meets requirements
- [ ] Edge cases handled
- [ ] Error conditions handled
- [ ] Input validation implemented
- [ ] Output format correct

### Design
- [ ] Follows SOLID principles
- [ ] Appropriate design patterns used
- [ ] Proper separation of concerns
- [ ] Dependencies injected
- [ ] Interfaces used appropriately

### Code Quality
- [ ] Code is readable and self-documenting
- [ ] Naming is clear and consistent
- [ ] Functions are focused and small
- [ ] DRY principle followed
- [ ] Comments explain "why", not "what"

### Security
- [ ] Input sanitized and validated
- [ ] No secrets in code
- [ ] Error messages don't leak information
- [ ] Authentication/authorization implemented
- [ ] SQL injection prevention

### Performance
- [ ] Efficient algorithms used
- [ ] Database queries optimized
- [ ] Caching implemented where appropriate
- [ ] Async/await used correctly
- [ ] Memory usage considered

### Testing
- [ ] Unit tests written
- [ ] Integration tests where needed
- [ ] Edge cases tested
- [ ] Error conditions tested
- [ ] Test coverage adequate

### Documentation
- [ ] Public APIs documented
- [ ] Complex logic explained
- [ ] README updated if needed
- [ ] API documentation updated
- [ ] Architecture decisions recorded
