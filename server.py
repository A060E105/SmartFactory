import socket
import audio
import pickle
from A_weighting import A_weighting
from specgram import convert_to_specgram
from AI_analysis import AI_analysis


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
                        filename = client.recv(1024)
                        filename = filename.decode()
                        print (f'client data: {filename}')
                        if not filename:
                            break
                        audio_temp = audio.record()
                        AW_temp = A_weighting(48000, audio_temp)
                        audio.save_wav(filename, AW_temp)
                        convert_to_specgram(filename)
                        result_list = AI_analysis(filename)
                        result = pickle.dumps(result_list)
                        client.sendall(result)
                    except:
                        pass
        except:
            pass