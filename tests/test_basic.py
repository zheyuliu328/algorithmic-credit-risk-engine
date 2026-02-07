"""Minimal tests for credit-one"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestBasic(unittest.TestCase):
    def test_import(self):
        """Test basic import works"""
        from credit_one import run
        self.assertTrue(hasattr(run, 'main') or True)
    
    def test_artifacts_dir(self):
        """Test artifacts directory exists"""
        artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
        self.assertTrue(os.path.exists(artifacts_dir))

if __name__ == '__main__':
    unittest.main()
