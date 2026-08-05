def pytest_configure(config):
    config.addinivalue_line(
        "markers", "endpoint: transport tests (import api -> heavy core.init)")
    config.addinivalue_line(
        "markers", "model: needs the model bundle loaded via core.init (slow)")
