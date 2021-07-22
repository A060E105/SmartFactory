"""
    exapmle python socket
    client
"""

import socket
import pickle

host = '127.0.0.1'
port = 7001

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((host, port))

    while True:
        try:
            outdata = ''
            device = ''
            while not outdata:
                outdata = input('please input filename: ')
                if not outdata:
                    print("Warning: file name can't empty")

            while not device:
                device = input('please input device (1 or 2): ')
                if not device in ['1', '2']:
                    device = ''
            print(f'send: {outdata}, device: {device}')

            request = pickle.dumps([outdata, device])
            client.send(request)

            indata = pickle.loads(client.recv(4096))

            if not indata:
                print('server closed connection.')
                break
            print(indata)
            break
        except KeyboardInterrupt:
            client.close()