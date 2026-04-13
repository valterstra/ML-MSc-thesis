"""
Detached launcher for step_11c_ddqn_discharge.py.
Uses Windows DETACHED_PROCESS flag so training survives bash shell exit.
Run this once; it spawns the trainer and exits immediately.
"""
import os
import subprocess
import sys

DETACHED_PROCESS    = 0x00000008
CREATE_NO_WINDOW    = 0x08000000
CREATE_NEW_PROC_GRP = 0x00000200

BASE_DIR   = r'C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis\CareAI'
TORCH_LIBS = (r'C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis'
              r'\.venv\Lib\site-packages\torch\lib')
PYTHON_EXE = (r'C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis'
              r'\.venv\Scripts\python.exe')
LOG_FILE   = os.path.join(BASE_DIR, 'logs', 'legacy', 'icu_readmit', 'step_29_launch_ddqn_tier2_discharge_legacy.log')

ARGS = sys.argv[1:]   # forward any extra args (e.g. --smoke)

training_code = f"""
import os, sys
os.add_dll_directory(r'{TORCH_LIBS}')
import torch
sys.argv = ['step_11c'] + {ARGS!r}
exec(open(r'{BASE_DIR}\\\\scripts\\\\icu_readmit\\\\legacy\\\\step_27_ddqn_tier2_discharge_legacy.py').read())
"""

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as log_fh:
    proc = subprocess.Popen(
        [PYTHON_EXE, '-c', training_code],
        stdout=log_fh,
        stderr=log_fh,
        cwd=BASE_DIR,
        creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW | CREATE_NEW_PROC_GRP,
        close_fds=True,
    )

print(f"Training process launched. PID={proc.pid}")
print(f"Log: {LOG_FILE}")
print("Process is fully detached -- safe to close this shell.")
