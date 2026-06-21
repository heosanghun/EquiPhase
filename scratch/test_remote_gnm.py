import paramiko
import sys

def run_test():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        # We will write a test script inside container and run it
        container_cmd = (
            "docker exec f7b8ca33561a python -c \""
            "import os, time, numpy as np; "
            "t0 = time.time(); "
            "A = np.random.randn(1000, 1000); "
            "np.linalg.pinv(A); "
            "print('Default threads SVD time:', time.time() - t0); "
            "\""
        )
        stdin, stdout, stderr = client.exec_command(container_cmd)
        print(stdout.read().decode('utf-8'))
        
        container_cmd_1 = (
            "docker exec f7b8ca33561a python -c \""
            "import os; "
            "os.environ['OMP_NUM_THREADS'] = '1'; "
            "os.environ['MKL_NUM_THREADS'] = '1'; "
            "os.environ['OPENBLAS_NUM_THREADS'] = '1'; "
            "import time, numpy as np; "
            "t0 = time.time(); "
            "A = np.random.randn(1000, 1000); "
            "np.linalg.pinv(A); "
            "print('1 thread SVD time:', time.time() - t0); "
            "\""
        )
        stdin, stdout, stderr = client.exec_command(container_cmd_1)
        print(stdout.read().decode('utf-8'))
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
