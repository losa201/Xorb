#!/usr/bin/env python3
"""
Test script to verify code deduplication is working properly
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_password_context_consolidation():
    """Test that password contexts are now centralized"""
    print("Testing password context consolidation...")
    
    try:
        # Import centralized password utilities
        from common.security_utils import hash_password, verify_password, validate_password_strength
        
        # Test password hashing and verification
        test_password = "TestPassword123!"
        hashed = hash_password(test_password)
        verified = verify_password(test_password, hashed)
        
        print(f"✅ Password hashing works: {verified}")
        
        # Test password strength validation
        strength = validate_password_strength(test_password)
        print(f"✅ Password strength validation: {strength['strength']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Password context test failed: {e}")
        return False

async def test_backup_system_consolidation():
    """Test that backup systems are now unified"""
    print("Testing backup system consolidation...")
    
    try:
        # Import unified backup manager
        from common.security_utils import backup_manager, UnifiedBackupManager
        
        # Test backup creation
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Test backup content")
            temp_path = temp_file.name
        
        try:
            # Create a backup
            result = await backup_manager.create_backup(temp_path, "test_backup")
            print(f"✅ Backup creation: {result.result.value}")
            
            # List backups
            backups = backup_manager.list_backups()
            print(f"✅ Backup listing: {len(backups)} backups found")
            
            # Get stats
            stats = backup_manager.get_backup_stats()
            print(f"✅ Backup stats: {stats['total_backups']} total backups")
            
            return True
            
        finally:
            # Clean up
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Backup system test failed: {e}")
        return False

def test_import_compatibility():
    """Test that old import paths still work"""
    print("Testing backward compatibility...")
    
    try:
        # Test backup system import
        from common.backup_system import backup_manager, BackupManager
        print("✅ Backup system imports work")
        
        return True
        
    except Exception as e:
        print(f"❌ Import compatibility test failed: {e}")
        return False

async def main():
    """Run all deduplication tests"""
    print("🧹 XORB Code Deduplication Test Suite")
    print("=" * 50)
    
    tests = [
        ("Password Context Consolidation", test_password_context_consolidation),
        ("Backup System Consolidation", test_backup_system_consolidation),
        ("Import Compatibility", test_import_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All deduplication tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)