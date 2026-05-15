import logging
from pathlib import Path

def pytest_configure(config):
    """
    Configure logging for the test suite.
    All logs from the application and tests will be written to the ./logs directory.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure the root logger to write to a file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "test_run.log"),
            logging.StreamHandler()
        ]
    )

    # Set specific log levels for project modules to ensure we see deep research logs
    logging.getLogger("src.agents.deep_research").setLevel(logging.DEBUG)
    logging.getLogger("src.agents").setLevel(logging.INFO)

def pytest_runtest_setup(item):
    """
    Clear the log file before each test run to avoid confusion.
    """
    log_file = Path("logs/test_run.log")
    if log_file.exists():
        # Open in write mode to clear it
        with open(log_file, 'w') as f:
            f.write("--- Test Run Started ---\n")
