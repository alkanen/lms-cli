from pathlib import Path
import subprocess
import sys


def run_test(_context: dict, test_path: str) -> str:
    """
    Run a pytest test file and capture the output.

    Args:
        _context (dict): Unused context parameter.
        test_path (str): Path to the Python test file.

    Returns:
        str: Combined output of stdout and stderr from the subprocess.
    """

    print(f"In run_test({test_path=})")

    # Helper function
    def run_pytest_test(python_executable_path: str, test_file_path: str) -> str:
        """
        Run a pytest test file using the specified pytest executable and capture the output.

        Args:
            python_executable_path (str): Path to the python executable.
            test_file_path (str): Path to the Python test file.

        Returns:
            str: Combined output of stdout and stderr from the subprocess.
        """
        try:
            result = subprocess.run(
                [python_executable_path, "-m", "pytest", test_file_path, "-v"],
                capture_output=True,
                text=True,
                check=False,
            )
            combined_output = result.stdout + result.stderr
            return combined_output
        except Exception as e:
            return f"Error running pytest: {str(e)}"

    # Use pytest from the current Python environment
    print(f"Running test: {test_path}")
    output = run_pytest_test(sys.executable, test_path)
    print(f"Output: {output}")

    return output
