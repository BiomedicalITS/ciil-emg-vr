import socket
import subprocess
import json
HOST = 'localhost'  # Host IP address
PORT = 12348       # Port number to listen on

def StartServer():
    flag = True
    #sgt=None
    try:
        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind the socket to a specific address and port
        server_socket.bind((HOST, PORT))
        # Listen for incoming connections
        server_socket.listen(1)
        print(f"Server started on {HOST}:{PORT}")
        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address[0]}:{client_address[1]}")
        
        while flag:
            try:
                # Receive data from Unity
                data = client_socket.recv(1024).decode('utf-8')[-1]
                data = int(data)

                ProcessReceivedData(data)
            except Exception as e:
                print(f"Error processing data: {e}")
                client_socket, client_address = server_socket.accept()
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        client_socket.close()
        server_socket.close()

def ProcessReceivedData(data):
    print("Received data from Unity:", data)
    if data == 1:
        proc.stdin.write(b'-cmd 11\n')
        proc.stdin.flush()
        ret = proc.stdout.readline().decode()
        dat = json.loads(ret)
        print(dat)
    if data == 0:
        proc.stdin.write(b'-cmd 12\n')
        proc.stdin.flush()
        ret = proc.stdout.readline().decode()
        dat = json.loads(ret)
        print(dat)


if (__name__ == "__main__"):
    proc = subprocess.Popen(['C:/Users/Thoma/OneDrive/Bureau/New folder (2)/sifi_bridge'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # To avoid deadlocks: careful to: add \n to output, flush output, use
    # readline() rather than read()

    # Connect to device
    connected = False
    while not connected:
        proc.stdin.write(b'-c BioPoint_v1_2\n')
        proc.stdin.flush()

        ret = proc.stdout.readline().decode()

        dat = json.loads(ret)

        print(dat)

        if dat["connected"] == 1:
            connected = True
        else:
            print("Could not connect. Retrying.")
        
    StartServer()
    """
    # Setup channels
    proc.stdin.write(b'-s ch (0,1,0,0,0) -cmd 0\n')
    proc.stdin.flush()

    # Start experiment
    while True:
        # Read data in
        ret = proc.stdout.readline().decode()
        print(json.loads(ret))
    """
