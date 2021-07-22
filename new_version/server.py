import socket
import SmartFactoryService as SFS
from SmartFactoryService import Audio
import pickle
import threading
from multiprocessing import Queue, Process

HOST = ''
PORT_1 = 7000
PORT_2 = 7001

devices = {'1': Audio.DEVICE_1, '2': Audio.DEVICE_2}

class Service(Process):
    def __init__(self, config: tuple) -> None:
        Process.__init__(self)
        self.HOST, self.PORT = config
        self.name = 'one server' if self.PORT == 7000 else 'two server'
        print(f"create {self.name}")


    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            # server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.HOST, self.PORT))
            server.setblocking(False)
            server.settimeout(10)
            server.listen(2)

            while True:
                try:
                    client, addr = server.accept()

                    print(f"connection: {addr}")

                    with client:
                        while True:
                            try:
                                queue = Queue()
                                request = pickle.loads(client.recv(1024))
                                filename = request[0]
                                device = devices[request[1]]
                                print(self.name)
                                print(f'client data: {filename}, device: {device}')
                                if not filename:
                                    break
                                sfs = SFS.SmartFactoryService(filename, device=device, model='./model.h5', queue=queue)
                                sfs.run()
                                result_list = queue.get()
                                result = pickle.dumps(result_list)
                                client.sendall(result)
                                client.close()
                                break
                            except socket.timeout:
                                client.close()
                                print('time out')
                                break
                except:
                    pass

if __name__ == '__main__':
    print("start program")
    Service((HOST, PORT_1)).start()
    Service((HOST, PORT_2)).start()
