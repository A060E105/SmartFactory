import socket


HOST = ''
PORT = 7000


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    server.setblocking(False)
    server.listen(5)

    print ('start program')

    while True:
        try:
            client, addr = server.accept()
            print (f'connection by: {addr}')

            with client:
                while True:
                    try:
                        data = client.recv(1024)
                        print (f'client data: {data}')
                        if not data:
                            break
                        client.sendall(data)
                    except:
                        pass
        except:
            pass