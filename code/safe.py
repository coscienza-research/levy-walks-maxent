"""Resource guard: import this module FIRST (before numpy) in every heavy script.

- single-threaded numerical libraries (avoids oversubscription with process pools)
- below-normal process priority on Windows (inherited by child processes), so the
  machine stays responsive
- MAX_WORKERS: at most half of the logical cores, and never more than 3
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

PRIORITY_SET = False
if os.name == "nt":
    try:
        import ctypes
        from ctypes import wintypes
        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _k32.GetCurrentProcess.restype = wintypes.HANDLE
        _k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        _k32.SetPriorityClass.restype = wintypes.BOOL
        PRIORITY_SET = bool(_k32.SetPriorityClass(_k32.GetCurrentProcess(), 0x00004000))  # BELOW_NORMAL
    except Exception:
        PRIORITY_SET = False

MAX_WORKERS = max(1, min(3, (os.cpu_count() or 2) // 2))
