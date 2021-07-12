"""
    exapmle python socket
    client
"""

import socket
import pickle

host = '127.0.0.1'
port = 7000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((host, port))

    while True:
        try:
            outdata = input('please input filename: ')
            print (f'send: {outdata}')

            client.send(outdata.encode())

            # indata = client.recv(1024)
            indata = pickle.loads(client.recv(4096))

            if not indata:
                print ('server closed connection.')
                break
            # print (f'recv: {indata.decode()}')
            print (indata)
        except KeyboardInterrupt:
            client.close()