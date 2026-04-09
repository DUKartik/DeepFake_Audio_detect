"""pytest.ini / conftest.py-style configuration"""
# conftest.py
import os
import pytest

# Ensure all environment variables are set before any module import
def pytest_configure(config):
    defaults = {
        "WA_TOKEN": "test_token",
        "PHONE_NUMBER_ID": "12345",
        "VERIFY_TOKEN": "test_verify",
        "APP_SECRET": "test_secret",
        "REDIS_URL": "redis://localhost:6379/0",
        "R2_BUCKET": "test-bucket",
        "R2_ENDPOINT_URL": "",
        "R2_ACCESS_KEY_ID": "",
        "R2_SECRET_ACCESS_KEY": "",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "testpass",
        "USE_GPU": "false",
        "AASIST_MODEL_PATH": "./weights/AASIST.pth",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)
