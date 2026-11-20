

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires the local database and/or model runtime")
    config.addinivalue_line("markers", "safe_integration: offline HTTP integration test with mocked dependencies")
