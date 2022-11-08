import os
import multiprocessing
import sys

MP_START_MODE = "spawn"
MP_FORCE = False 

g_setup_start_mode = None 
def setup_mp_start_mode():
    global g_setup_start_mode 
    if g_setup_start_mode:
        return 

    pid = os.getpid()    
    print(f"'SMSM': {sys.argv} @pid:{pid}")
    try:
        mode = multiprocessing.get_start_method()
        if mode != MP_START_MODE:
            msg = f"'SMSM': try to set_start_mode from '{mode}' to '{MP_START_MODE}' @ pid:{pid} "
            print(msg)
            multiprocessing.set_start_method(MP_START_MODE, force=MP_FORCE)
    except Exception as ex:
        try:
            msg = f"'SMSM': exception occured {ex}, when setup start_method: {MP_START_MODE}"
            print(msg)
        except: # noqa
            pass
    g_setup_start_mode = True
