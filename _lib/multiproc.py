import multiprocessing

def run_multiproc(fun,arg,numProc):
    if numProc == 1:
        return map(fun,arg)
    p = multiprocessing.Pool(processes = numProc)
    try:
        result = p.map_async(fun,list(arg)).get(9999999)
        # result = p.map(fun,arg)
        p.close()
        p.join()
        return result
    except KeyboardInterrupt:
        p.terminate()
        print("Execution has been interrupted by the user.")
        return [[0,0]]
