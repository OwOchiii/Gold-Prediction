"""
Logging configuration module for Gold Price Prediction System.

This module sets up structured logging for all components of the system,
providing consistent log formatting and multiple output handlers.
"""

import logging
import sys
from pathlib import Path
from config import Config


def setup_logging(
    log_name: str = 'gold_prediction',
    log_level: str = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_name: Name of the logger (default: 'gold_prediction')
        log_level: Logging level (default: from Config.LOG_LEVEL)
        log_to_file: Whether to log to file (default: True)
        log_to_console: Whether to log to console (default: True)
    
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(log_name)
    
    # Set log level
    if log_level is None:
        log_level = Config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_file = Config.get_log_path(f'{log_name}.log')
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    If the logger doesn't exist, it will be created with default configuration.
    
    Args:
        name: Name of the logger (typically __name__ of the module)
    
    Returns:
        Logger instance
    """
    # Check if logger already exists and is configured
    logger = logging.getLogger(name)
    
    # If no handlers, set up default configuration
    if not logger.handlers:
        return setup_logging(log_name=name)
    
    return logger


# Default logger for the application
default_logger = setup_logging()


if __name__ == '__main__':
    # Test logging configuration
    test_logger = get_logger('test_logger')
    
    test_logger.debug('This is a debug message')
    test_logger.info('This is an info message')
    test_logger.warning('This is a warning message')
    test_logger.error('This is an error message')
    test_logger.critical('This is a critical message')
    
    print(f"\nLog file created at: {Config.get_log_path('test_logger.log')}")
