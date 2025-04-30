# _PythonProjectTemplate/tests/test_example.py
import pytest # We need to make sure pytest is in requirements.txt

def test_always_passes():
    """
    This is a basic placeholder test.
    When you run tests, this one should always pass.
    Replace this with real tests for your project's code.
    """
    assert True == True # A simple assertion that is always true

# You can add more tests here as needed
# For example, if you had a function 'add(x, y)' in main.py:
#
# from ..main import add # Use '..' to import from parent directory
#
# def test_add_function():
#    assert add(2, 3) == 5