"""Unit tests for authentication functionality."""

import pytest
from unittest.mock import AsyncMock, patch
from src.api.app.services.auth_security_service import AuthSecurityService
from src.api.app.domain.exceptions import DomainException


class TestAuthSecurityService:
    """Test cases for AuthSecurityService."""

    @pytest.fixture
    def auth_service(self, mock_database, mock_redis):
        """Create AuthSecurityService instance for testing."""
        return AuthSecurityService(
            user_repository=mock_database,
            redis_client=mock_redis
        )

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service, sample_user_data):
        """Test successful user authentication."""
        # Mock user repository response
        auth_service.user_repository.get_user_by_username.return_value = sample_user_data
        
        with patch.object(auth_service, 'verify_password', return_value=True):
            user = await auth_service.authenticate_user('testuser', 'password123')
            assert user['username'] == 'testuser'
            assert user['email'] == 'test@example.com'

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_credentials(self, auth_service):
        """Test authentication with invalid credentials."""
        auth_service.user_repository.get_user_by_username.return_value = None
        
        with pytest.raises(DomainException, match="Invalid credentials"):
            await auth_service.authenticate_user('baduser', 'badpassword')

    @pytest.mark.asyncio
    async def test_create_access_token(self, auth_service, sample_user_data):
        """Test JWT token creation."""
        with patch('src.api.app.services.auth_security_service.jwt.encode') as mock_encode:
            mock_encode.return_value = 'test.jwt.token'
            
            token = await auth_service.create_access_token(sample_user_data)
            assert token == 'test.jwt.token'
            mock_encode.assert_called_once()

    def test_verify_password(self, auth_service):
        """Test password verification."""
        # Mock password hash verification
        with patch('src.api.app.services.auth_security_service.pwd_context.verify') as mock_verify:
            mock_verify.return_value = True
            
            result = auth_service.verify_password('plaintext', 'hashed')
            assert result is True
            mock_verify.assert_called_once_with('plaintext', 'hashed')

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        with patch('src.api.app.services.auth_security_service.pwd_context.hash') as mock_hash:
            mock_hash.return_value = 'hashed_password'
            
            result = auth_service.hash_password('plaintext')
            assert result == 'hashed_password'
            mock_hash.assert_called_once_with('plaintext')

    @pytest.mark.asyncio
    async def test_validate_token_success(self, auth_service):
        """Test successful token validation."""
        with patch('src.api.app.services.auth_security_service.jwt.decode') as mock_decode:
            mock_decode.return_value = {'sub': 'testuser', 'exp': 9999999999}
            
            result = await auth_service.validate_token('valid.jwt.token')
            assert result['sub'] == 'testuser'

    @pytest.mark.asyncio
    async def test_validate_token_expired(self, auth_service):
        """Test token validation with expired token."""
        with patch('src.api.app.services.auth_security_service.jwt.decode') as mock_decode:
            from jwt.exceptions import ExpiredSignatureError
            mock_decode.side_effect = ExpiredSignatureError()
            
            with pytest.raises(DomainException, match="Token has expired"):
                await auth_service.validate_token('expired.jwt.token')

    @pytest.mark.asyncio
    async def test_rate_limiting(self, auth_service, mock_redis):
        """Test rate limiting functionality."""
        # Mock Redis rate limiting
        mock_redis.get.return_value = b'5'  # 5 attempts
        mock_redis.ttl.return_value = 3600   # 1 hour remaining
        
        with pytest.raises(DomainException, match="Too many login attempts"):
            await auth_service.check_rate_limit('192.168.1.1')

    @pytest.mark.asyncio
    async def test_mfa_token_generation(self, auth_service, sample_user_data):
        """Test MFA token generation."""
        with patch('src.api.app.services.auth_security_service.pyotp.TOTP') as mock_totp:
            mock_totp.return_value.provisioning_uri.return_value = 'otpauth://...'
            
            result = await auth_service.generate_mfa_secret(sample_user_data)
            assert 'secret' in result
            assert 'qr_code_url' in result