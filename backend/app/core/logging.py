"""Structured logging — plaintext format for stdout / Cloud Logging compatibility."""
import logging, json, sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
