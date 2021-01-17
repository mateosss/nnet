from time import time

last_time = time()
def lap(msg):
    global last_time
    current_time = time()
    print(f" [{current_time - last_time:.6f}s]\n{msg}", end="")
    last_time = current_time
