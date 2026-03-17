"""
Tests for the placeholder encryption production guard.
"""

import os
import pytest

from utils import encrypt_data, decrypt_data, _check_production_guard


class TestEncryptionProductionGuard:
    """Tests that placeholder encryption refuses to run in production."""

    def test_encrypt_works_in_development(self):
        """Placeholder encryption should work when APP_ENV is not production."""
        # Default env is "development" in tests
        result = encrypt_data(b"hello", b"key")
        assert result.endswith(b"_placeholder_not_encrypted")

    def test_decrypt_works_in_development(self):
        """Placeholder decryption should work when APP_ENV is not production."""
        encrypted = encrypt_data(b"hello", b"key")
        decrypted = decrypt_data(encrypted, b"key")
        assert decrypted == b"hello"

    def test_encrypt_refuses_in_production(self, monkeypatch):
        """Placeholder encryption should raise RuntimeError in production."""
        monkeypatch.setenv("APP_ENV", "production")
        with pytest.raises(RuntimeError, match="production"):
            encrypt_data(b"hello", b"key")

    def test_decrypt_refuses_in_production(self, monkeypatch):
        """Placeholder decryption should raise RuntimeError in production."""
        monkeypatch.setenv("APP_ENV", "production")
        with pytest.raises(RuntimeError, match="production"):
            decrypt_data(b"hello_placeholder_not_encrypted", b"key")

    def test_production_guard_directly(self, monkeypatch):
        """_check_production_guard should raise in production."""
        monkeypatch.setenv("APP_ENV", "production")
        with pytest.raises(RuntimeError, match="Placeholder encryption cannot be used"):
            _check_production_guard("encryption")

    def test_production_guard_passes_in_dev(self):
        """_check_production_guard should not raise in development."""
        # Should not raise
        _check_production_guard("encryption")

    def test_decrypt_invalid_data_raises(self):
        """Decrypting data without the placeholder suffix should raise ValueError."""
        with pytest.raises(ValueError, match="not encrypted with the placeholder"):
            decrypt_data(b"not_encrypted_data", b"key")

    def test_roundtrip(self):
        """Encrypt then decrypt should return original data."""
        original = b"test secret data 12345"
        encrypted = encrypt_data(original, b"any_key")
        decrypted = decrypt_data(encrypted, b"any_key")
        assert decrypted == original
