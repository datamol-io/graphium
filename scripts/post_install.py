import subprocess
import sys

# Use the current Python interpreter to run pip
subprocess.check_call(['ls'])
# sys.executable, '-m', 'pip', 'install', package_name