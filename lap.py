from time import time
start_time = 0

def lap(msg):
    global start_time
    if not start_time:
        start_time = time()
        print(msg, end=" ")
        return
    end_time = time()
    print(f"[{end_time - start_time:.6f}]")
    print(msg, end=" ")
    start_time = end_time
