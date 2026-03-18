import multiprocessing as mp
import tempfile
import os
import pandas as pd
import numpy as np
import run_hydrus as rh
import tempfile
import psutil


### Define Functions ###
def _worker(queue, vgs, calib, atm, atm_hydrus, obs, days, timesteps2, result_path):
    print("Worker started")
    try:
        alldata, hydrus_output = rh.run_dual_perm(vgs, calib, atm, atm_hydrus, obs, days, timesteps2)
        # Save results to temp files instead of passing through queue (avoids slow pickling)
        np.save(result_path + ".npy", hydrus_output)
        alldata.to_pickle(result_path + "_alldata.pkl")
        print("Hydrus finished")
        queue.put("success")
    except Exception as e:
        print("Worker error:", e)
        queue.put(e)


def run_with_timeout(vgs, calib, atm, atm_hydrus, obs, days, timesteps2, fallback, timeout=15):
    result_path = tempfile.NamedTemporaryFile(delete=False).name
    queue = mp.Queue()
    p = mp.Process(
        target=_worker,
        args=(queue, vgs, calib, atm, atm_hydrus, obs, days, timesteps2, result_path)
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        print("TIMEOUT OCCURRED")
        # Kill entire process tree including H1D_Dual.exe
        try:
            parent = psutil.Process(p.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        p.join()
        return (None, fallback)

    print("Process finished, reading result from file")
    status = queue.get()

    if isinstance(status, Exception):
        raise status

    # Read results back from temp files
    hydrus_output = np.load(result_path + ".npy")
    alldata = pd.read_pickle(result_path + "_alldata.pkl")

    # Clean up temp files
    os.remove(result_path + ".npy")
    os.remove(result_path + "_alldata.pkl")

    return alldata, hydrus_output

