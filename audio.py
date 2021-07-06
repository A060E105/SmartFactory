import wave
from pyaudio import PyAudio, paInt16
import tqdm

source_path = './source/'

# recording
def record():
	pa = PyAudio()
	stream = pa.open(format=paInt16,
	 channels=1, 
	 rate=48000, 
	 input=True, 
	 frames_per_buffer=2048)
	my_buf = []
	for _ in tqdm.trange(50):	# record  time
		string_audio_data = stream.read(2048)
		my_buf.append(string_audio_data)
	stream.close()
	return my_buf

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
	wf.writeframes(b"".join(data))
	wf.close()