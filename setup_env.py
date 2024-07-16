import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import subprocess
import os

# zsh script that sources environment and sets env variables
env_output = subprocess.check_output(['zsh', '-c', 'source setup_env_var.zsh && env'])

# Update the Python process environment
for line in env_output.decode().split('\n'):
    if line and '=' in line:
        key, value = line.split('=', 1)
        os.environ[key] = value