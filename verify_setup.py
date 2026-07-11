"""
Verification script for project setup.

This script validates that all required components are properly configured:
- Directory structure
- Configuration parameters
- Logging system
- Package imports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config
from src.logger import get_logger


def verify_directories():
    """Verify all required directories exist."""
    print("=" * 60)
    print("DIRECTORY STRUCTURE VERIFICATION")
    print("=" * 60)
    
    directories = [
        ('Data Directory', Config.DATA_DIR),
        ('Models Directory', Config.MODEL_DIR),
        ('Reports Directory', Config.REPORTS_DIR),
        ('Logs Directory', Config.LOGS_DIR),
        ('Source Directory', Config.BASE_DIR / 'src'),
        ('Tests Directory', Config.BASE_DIR / 'tests'),
    ]
    
    all_exist = True
    for name, path in directories:
        exists = path.exists() and path.is_dir()
        status = "✓ OK" if exists else "✗ MISSING"
        print(f"{status:10} {name:25} {path}")
        all_exist = all_exist and exists
    
    return all_exist


def verify_configuration():
    """Verify configuration parameters are set."""
    print("\n" + "=" * 60)
    print("CONFIGURATION VERIFICATION")
    print("=" * 60)
    
    params = [
        ('Sequence Length', Config.SEQUENCE_LENGTH),
        ('Train/Val/Test Ratio', f"{Config.TRAIN_RATIO}/{Config.VAL_RATIO}/{Config.TEST_RATIO}"),
        ('Batch Size', Config.BATCH_SIZE),
        ('Epochs', Config.EPOCHS),
        ('Target Column', Config.TARGET_COLUMN),
        ('Lag Periods', Config.LAG_PERIODS),
        ('Rolling Windows', Config.ROLLING_WINDOWS),
        ('Economic Indicators', list(Config.INDICATORS.keys())),
    ]
    
    for name, value in params:
        print(f"  {name:25} {value}")
    
    # Verify ratios sum to 1.0
    ratio_sum = Config.TRAIN_RATIO + Config.VAL_RATIO + Config.TEST_RATIO
    ratio_ok = abs(ratio_sum - 1.0) < 0.001
    
    print(f"\n  Ratio Sum Check:          {'✓ OK' if ratio_ok else '✗ FAIL'} (sum={ratio_sum})")
    
    return ratio_ok


def verify_logging():
    """Verify logging system is working."""
    print("\n" + "=" * 60)
    print("LOGGING SYSTEM VERIFICATION")
    print("=" * 60)
    
    try:
        # Create test logger
        logger = get_logger('setup_verification')
        logger.info("Testing logging system")
        
        # Check log file was created
        log_file = Config.get_log_path('setup_verification.log')
        log_exists = log_file.exists()
        
        print(f"  Logger Creation:          ✓ OK")
        print(f"  Log File Created:         {'✓ OK' if log_exists else '✗ FAIL'}")
        print(f"  Log File Path:            {log_file}")
        
        return True
    except Exception as e:
        print(f"  Logger Creation:          ✗ FAIL - {e}")
        return False


def verify_imports():
    """Verify package imports work correctly."""
    print("\n" + "=" * 60)
    print("PACKAGE IMPORTS VERIFICATION")
    print("=" * 60)
    
    imports = [
        ('config', 'Config'),
        ('src', '__version__'),
        ('src.logger', 'get_logger'),
    ]
    
    all_ok = True
    for module, attr in imports:
        try:
            exec(f"from {module} import {attr}")
            print(f"  ✓ OK     import {attr} from {module}")
        except ImportError as e:
            print(f"  ✗ FAIL   import {attr} from {module} - {e}")
            all_ok = False
    
    return all_ok


def verify_required_files():
    """Verify required files exist."""
    print("\n" + "=" * 60)
    print("REQUIRED FILES VERIFICATION")
    print("=" * 60)
    
    files = [
        ('config.py', Config.BASE_DIR / 'config.py'),
        ('requirements.txt', Config.BASE_DIR / 'requirements.txt'),
        ('README.md', Config.BASE_DIR / 'README.md'),
        ('src/__init__.py', Config.BASE_DIR / 'src' / '__init__.py'),
        ('src/logger.py', Config.BASE_DIR / 'src' / 'logger.py'),
        ('tests/__init__.py', Config.BASE_DIR / 'tests' / '__init__.py'),
    ]
    
    all_exist = True
    for name, path in files:
        exists = path.exists() and path.is_file()
        status = "✓ OK" if exists else "✗ MISSING"
        print(f"  {status:10} {name}")
        all_exist = all_exist and exists
    
    return all_exist


def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "GOLD PREDICTION SYSTEM SETUP VERIFICATION" + " " * 7 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        'Directories': verify_directories(),
        'Configuration': verify_configuration(),
        'Logging': verify_logging(),
        'Imports': verify_imports(),
        'Required Files': verify_required_files(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status:10} {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL CHECKS PASSED - Setup is complete!")
    else:
        print("  ✗ SOME CHECKS FAILED - Review errors above")
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
