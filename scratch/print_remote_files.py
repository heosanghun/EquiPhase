import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('203.255.93.75', port=10022, username='shoon', password='rvlab@guest_sh')
    
    print("=== Remote /TRACKING/shoon/EquiPhase/iss_module.py ===")
    stdin, stdout, stderr = client.exec_command('sed -n "135,160p" /TRACKING/shoon/EquiPhase/iss_module.py')
    print(stdout.read().decode())
    
    print("=== Remote /TRACKING/shoon/EquiPhase/equiphase/models/symplectic_deq.py ===")
    stdin, stdout, stderr = client.exec_command('sed -n "45,70p" /TRACKING/shoon/EquiPhase/equiphase/models/symplectic_deq.py')
    print(stdout.read().decode())
    
    client.close()

if __name__ == "__main__":
    main()
