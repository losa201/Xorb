"""
Tests for secure JWT secret management
"""

import os
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.api.app.core.secure_jwt import SecureJWTManager, SecurityError, JWTSecretMetadata


class TestSecureJWTManager:
    """Test cases for SecureJWTManager"""
    
    def test_secret_validation_length_requirement(self):
        """Test that secrets must meet minimum length requirement"""
        manager = SecureJWTManager(environment="development")
        
        # Too short
        assert not manager._validate_secret("short")
        assert not manager._validate_secret("a" * 32)  # 32 chars
        
        # Just right
        assert manager._validate_secret("a" * 64)  # 64 chars
        assert manager._validate_secret("a" * 128)  # 128 chars
    
    def test_entropy_calculation(self):
        """Test entropy calculation for secret strength"""
        manager = SecureJWTManager(environment="development")
        
        # Low entropy (all same character)
        low_entropy = "a" * 64
        entropy = manager._calculate_entropy(low_entropy)
        assert entropy == 0.0
        
        # High entropy (random mix)
        high_entropy = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@"
        entropy = manager._calculate_entropy(high_entropy)
        assert entropy > 5.0
    
    def test_weak_pattern_detection(self):
        """Test detection of weak patterns in secrets"""
        manager = SecureJWTManager(environment="development")
        
        # Sequential pattern
        sequential = "abcdefghijklmnopqrstuvwxyz" + "A" * 38  # 64 chars total
        assert manager._has_weak_patterns(sequential)
        
        # Repeated character pattern
        repeated = "aabbccdd" * 8  # 64 chars with low variety
        assert manager._has_weak_patterns(repeated)
        
        # Good pattern
        good_secret = "Kj8#mN2!pQ7$vX9&zL4@wB6*tY1%rE3^uI5+oA8-sD0="
        assert not manager._has_weak_patterns(good_secret)
    
    @patch.dict(os.environ, {"JWT_SECRET": "a" * 64}, clear=False)
    def test_load_from_environment_valid(self):
        """Test loading valid secret from environment"""
        manager = SecureJWTManager(environment="development")
        
        assert manager._current_secret == "a" * 64
        assert manager._secret_metadata.source == "env"
        assert manager._secret_metadata.length == 64
    
    @patch.dict(os.environ, {"JWT_SECRET": "short"}, clear=False)
    def test_load_from_environment_invalid(self):
        """Test handling of invalid environment secret"""
        with pytest.raises(SecurityError, match="JWT_SECRET does not meet security requirements"):
            SecureJWTManager(environment="production")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_no_secret_development(self):
        """Test handling when no secret is available in development"""
        manager = SecureJWTManager(environment="development")
        
        # Should generate temporary secret
        assert manager._current_secret is not None
        assert len(manager._current_secret) >= 64
        assert manager._secret_metadata.source == "generated"
    
    def test_secret_generation(self):
        """Test secure secret generation"""
        manager = SecureJWTManager(environment="development")
        
        # Generate multiple secrets to test consistency
        secrets = [manager._generate_secret() for _ in range(10)]
        
        for secret in secrets:
            assert len(secret) >= 64
            assert manager._validate_secret(secret)
        
        # All secrets should be unique
        assert len(set(secrets)) == len(secrets)
    
    def test_rotation_needed_check(self):
        """Test rotation timing logic"""
        manager = SecureJWTManager(environment="development")
        
        # New secret shouldn't need rotation
        assert not manager._needs_rotation()
        
        # Mock old secret
        manager._secret_metadata = JWTSecretMetadata(
            created_at=time.time() - 25 * 60 * 60,  # 25 hours ago
            entropy=5.5,
            length=64,
            rotation_count=1,
            source="vault"
        )
        
        assert manager._needs_rotation()
        
        # Environment secrets should never rotate
        manager._secret_metadata.source = "env"
        assert not manager._needs_rotation()
    
    def test_get_secret_info(self):
        """Test secret information reporting"""
        manager = SecureJWTManager(environment="development")
        
        info = manager.get_secret_info()
        
        assert "length" in info
        assert "entropy" in info
        assert "age_hours" in info
        assert "rotation_count" in info
        assert "source" in info
        
        assert info["length"] >= 64
        assert info["entropy"] >= 5.0
    
    def test_force_rotation(self):
        """Test forced secret rotation"""
        manager = SecureJWTManager(environment="development")
        
        old_secret = manager._current_secret
        old_rotation_count = manager._secret_metadata.rotation_count
        
        manager.force_rotation()
        
        # Secret should have changed
        assert manager._current_secret != old_secret
        assert manager._secret_metadata.rotation_count > old_rotation_count
    
    @patch('src.api.app.core.secure_jwt.time.time')
    def test_automatic_rotation_check(self, mock_time):
        """Test automatic rotation checking with throttling"""
        manager = SecureJWTManager(environment="development")
        
        # Mock time progression
        mock_time.side_effect = [
            1000.0,  # Initial time
            1300.0,  # 5 minutes later (should check)
            1310.0,  # 10 seconds later (should not check)
            1700.0,  # 5 minutes later again (should check)
        ]
        
        # First call - should check rotation
        secret1 = manager.get_signing_key()
        
        # Second call - should not check (throttled)
        secret2 = manager.get_signing_key()
        
        # Third call - should check again
        secret3 = manager.get_signing_key()
        
        # All should return the same secret (no rotation needed)
        assert secret1 == secret2 == secret3


if __name__ == "__main__":
    pytest.main([__file__])