import wave
from pyaudio import PyAudio, paInt16
import tqdm
import numpy as np
import os

source_path = './source/'

def my_mkdir( path ):
    now_path = ''
    for i in path.split('\\'):
        now_path = os.path.join( now_path , i )
        try:
            os.mkdir(now_path)
        except:
            pass


# recording
def record():
	my_mkdir(source_path)
	pa = PyAudio()
	stream = pa.open(format=paInt16,
	 channels=1, 
	 rate=48000, 
	 input=True, 
	 frames_per_buffer=2048)
	my_buf = []
	for _ in tqdm.trange(int(48000/ 2048 * 5)):	# record  time
		string_audio_data = stream.read(2048)
		my_buf.append(string_audio_data)
	stream.close()
	return np.array(my_buf).tobytes()

# show all input device
def getDevice():
	p = PyAudio()
	info = p.get_host_api_info_by_index(0)
	numdevice = info.get('deviceCount')
	for i in range(0, numdevice):
		print (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'))
		if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
			pass

# save data to wav file
def save_wav(filename, data):
	wf = wave.open(f"{source_path}{filename}.wav", "wb")
	wf.setnchannels(1)
	wf.setsampwidth(2)
	wf.setframerate(48000)
	wf.writeframes(data)
	wf.close()