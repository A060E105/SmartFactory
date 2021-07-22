"""
    Smart Factory
    audio class
"""

import re
import os
# from tensorflow.python.framework.ops import device
import tqdm
import wave
import numpy as np
from pyaudio import PyAudio, paInt16
# A-weighting
from numpy import pi, polymul
from scipy.signal import bilinear, lfilter
# convert to specgram
import io
import cv2
from PIL import Image
from numpy.lib import stride_tricks
from matplotlib import pyplot as plt
# AI analysis
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
# thread
from threading import Thread
from multiprocessing import Process, Queue

# disable debugging logs
# 0 -> all info
# 1 -> info message not print
# 2 -> info and warning message not print
# 3 -> info, warning and error message not print
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =======================================================
#                   public variable 
# =======================================================
SOURCE_PATH = './source/'
AUDIO_OUT_PATH = './audio/'
SPEC_PATH = './spec/'

# =======================================================
#                   public methods 
# =======================================================
def my_mkdir(path: str) -> None:
    now_path = ''
    for i in path.split('\\'):
        now_path = os.path.join(now_path, i)
        try:
            os.mkdir(now_path)
        except:
            pass

# =============================================================================
#   Audio class
# =============================================================================
class Audio:
    DEVICE_1 = r'Cotron EZM-001\)$'
    DEVICE_2 = r'Cotron EZM-001-2\)$'

    framerate = 48000
    # framerate = 96000
    samples = 4096
    sampwidth = 2
    channels = 1

    def __init__(self, filename, device='') -> None:
        self.filename = filename
        self.record_data = b''
        self.device = device

    def record(self) -> None:
        if self.hasDevice():
            pa = PyAudio()
            stream = pa.open(format=paInt16,
                channels=self.channels,
                rate=self.framerate,
                input=True,
                input_device_index=self.__getDevice(),
                frames_per_buffer=self.samples)
            my_buf = []
            for _ in tqdm.trange(int(self.framerate / self.samples * 6), desc=f"record {self.filename}.wav"):
                string_audio_data = stream.read(self.samples)
                my_buf.append(string_audio_data)
            stream.close()
            self.record_data = np.array(my_buf).tobytes()

    def save_wav(self) -> None:
        my_mkdir(SOURCE_PATH)
        wf = wave.open(f"{SOURCE_PATH}{self.filename}.wav", 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.sampwidth)
        wf.setframerate(self.framerate)
        wf.writeframes(self.record_data)
        wf.close()

    def A_weighting(self) -> None:
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        A1000 = 1.9997
    
        NUMs = [(2*pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
        DENs = polymul([1, 4*pi * f4, (2*pi * f4)**2],
                       [1, 4*pi * f1, (2*pi * f1)**2])
        DENs = polymul(polymul(DENs, [1, 2*pi * f3]),
                                     [1, 2*pi * f2])
    
        # Use the bilinear transformation to get the digital filter.
        # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
        b, a = bilinear(NUMs, DENs, self.framerate)
    
        data = np.frombuffer(self.record_data, dtype=np.short)
        y = lfilter(b, a, data)
    
        self.record_data = y.astype(np.short).tobytes()

    def getDeviceName(self) -> list:
        p = PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        device_name_list = []
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                device_name_list.append(p.get_device_info_by_host_api_device_index(0, i).get('name'))

        return device_name_list

    def __getDevice(self) -> "int | None":
        p = PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                if ((re.search(self.device, p.get_device_info_by_host_api_device_index(0, i).get('name'))) is not None):
                    return i

    def hasDevice(self) -> bool:
        if self.__getDevice() is None:
            # return False
            # develop test
            print ("has not device")
            return True
        else:
            print ("has device")
            return True


# =============================================================================
#       Specgram class
# =============================================================================
class Specgram():
    picture_width , picture_height = 200 , 100 #圖片長寬比
    CutTimeDef  = 2 #以1s截斷檔案
    SpaceNumDef = 1 #每次取時間間隔
    freq_split_list = [ [0,6000] ]

    def __init__(self, filename: str, cut=True, save_audio=False) -> None:
        self.filename = filename
        self.with_cut_file = cut
        self.save_split_audio = save_audio

    def toSpecgram(self):
        my_mkdir(AUDIO_OUT_PATH)
        my_mkdir(SPEC_PATH)
        index = 0
        for audio_info , image_list in self.__CutFile():
            index += 1
            if self.save_split_audio:
                sampwidth = audio_info[0]
                framerate = audio_info[1]
                temp_dataTemp = audio_info[2]
                audio_file_name = '{}_{}.wav'.format(self.filename, index)
                audio_save_dir = os.path.join( AUDIO_OUT_PATH, self.filename )
                my_mkdir(audio_save_dir)
                audio_save_path =  os.path.join( audio_save_dir , audio_file_name )  #設定音檔儲存路徑
                f = wave.open( audio_save_path, "wb")
                f.setnchannels(1) #固定輸出單聲道
                f.setsampwidth(sampwidth)
                f.setframerate(framerate)
                    # 將wav_data轉換為二進位制資料寫入檔案
                f.writeframes(temp_dataTemp.tobytes() )
                f.close()

            for freq_info,image in zip(self.freq_split_list , image_list):
                img_bin = Image.fromarray( image.astype( np.uint8 ), 'RGB')
                freq_str = '{}~{}'.format( freq_info[0], freq_info[1] )
                image_dir = os.path.join( SPEC_PATH, freq_str, self.filename, 'OK' )
                file_name = '{}_{}.png'.format( self.filename, index )
                my_mkdir(image_dir)
                image_path = os.path.join( image_dir, file_name)  #設定圖片儲存路徑
                img_bin.save( image_path )

    @property
    def file_path(self) -> str:
        return f"{SOURCE_PATH}{self.filename}.wav"

    def __CutFile(self) -> list:
        f = wave.open( self.file_path , "rb")
        params = f.getparams()
        
        nchannels, sampwidth, framerate, nframes = params[:4]
        CutFrameNum = framerate* self.CutTimeDef
        str_data = f.readframes(nframes)
        f.close()# 將波形資料轉換成陣列
        wave_data = np.frombuffer(str_data, dtype=np.short)

        if nchannels >= 2:  #轉換成雙聲道
            wave_data.shape = -1, 2
            wave_data = wave_data.T
        else:
            wave_data = wave_data.reshape( 1,wave_data.shape[0] )
        
        if self.with_cut_file:
            ptr_start = 0
            time = 0
            total_time = (wave_data.shape[1] / framerate / self.SpaceNumDef) - self.CutTimeDef #推測總共圖片數量
            total_time = int(total_time) + 1
            with tqdm.tqdm(total = total_time, desc=f"{self.filename}.wav to specgram") as pbar:
                while 1:
                    ptr_end = ptr_start + self.CutTimeDef * framerate
                    ptr_end = int(ptr_end)
                    if ptr_end <= nframes:
                        temp_dataTemp = wave_data[0][ptr_start:ptr_end]  #分割音檔
                        image_list = self.__plotstft( self.freq_split_list , self.picture_width , self.picture_height , framerate , temp_dataTemp ) #轉換成圖片
                        ptr_start += self.SpaceNumDef * framerate
                        ptr_start = int(ptr_start)
                        time += 1
                        pbar.update(1)
                        #print('\n' , ptr_start , ptr_end, '\n')
                        yield [sampwidth , framerate , temp_dataTemp] , image_list
                    else:
                        break
        else:
            temp_dataTemp = wave_data[0]
            image_list = self.__plotstft( self.freq_split_list , self.picture_width , self.picture_height , framerate , temp_dataTemp ) #轉換成圖片
            yield [sampwidth , framerate , temp_dataTemp] , image_list

    def __plotstft(self, freq_split_list, im_width, im_height, samplerate, samples, binsize=2**10, plotpath=None, colormap="jet") -> np:
        s = self.__stft(samples, binsize)
        sshow_origin, freq = self.__logscale_spec(s, factor=1.0, sr=samplerate)
        image_list = []
        for f_split in freq_split_list:
            freq = np.array(freq)
            mask = (freq >= f_split[0] ) * (freq < f_split[1] ) 
            index_list = [ index for index,x in enumerate(mask) if x]  #取出需要頻率的index
            sshow = sshow_origin[ : , index_list]
            
            # ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
            ims = 20. * np.log10(np.abs(sshow) / 10e+6)  #dBFS
            # ims = 20.*np.log10(np.abs(sshow)/10e-3)
            # ims = 20.*np.log10(np.abs(sshow)/32768)
            # ims = 20*np.log10(np.abs(sshow)/32768)
            # ims = np.abs(sshow) #STFT原始數值
            
            # timebins, freqbins = np.shape(ims)
            #print("timebins: ", timebins)
            #print("freqbins: ", freqbins)
            # plt.imshow(np.transpose(ims), origin="None", aspect="auto", cmap="jet", extent = None, interpolation='None', vmin= -160, vmax= 0)
            plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap="jet", extent = None, interpolation='None', vmin= -160, vmax= 0)
            plt.axis('off') 
            fig = plt.gcf()
            fig.set_size_inches(  im_width , im_height  ) #dpi = 300, output = 700*700 pixels
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            # 去除空白code https://blog.csdn.net/qq_30708445/article/details/103190850?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-1
            img_arr = self.__get_img_from_fig( fig , dpi = 1 )
            image_list.append( img_arr )
            plt.clf()
        return np.array(image_list)

    def __logscale_spec(self, spec, sr=96000, factor=20.) -> tuple:
        timebins, freqbins = np.shape(spec)
    
        scale = np.linspace(0, 1, freqbins) ** factor
        scale *= (freqbins-1)/max(scale)
        scale = np.unique(np.round(scale))
    
        # create spectrogram with new freq bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        for i in range(0, len(scale)):        
            if i == len(scale)-1:
                newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
            else:        
                newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)
    
        # list center freq of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
        freqs = []
        for i in range(0, len(scale)):
            if i == len(scale)-1:
                freqs += [np.mean(allfreqs[int(scale[i]):])]
            else:
                freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]
    
        return newspec, freqs

    def __stft(self, sig, frameSize, overlapFac=0.5, window=np.hanning) -> np:
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))
        # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
        samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
        # cols for windowing
        cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))
        
        frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
        frames *= win
        return np.fft.rfft(frames)    

    def __get_img_from_fig(self, fig, dpi=180) -> cv2:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

# =============================================================================
#   AI analysis class
# =============================================================================
class AI_analysis():
    resize_shape = ( 100, 200 )
    batch_size = 32
    my_class = ['NG', 'OK']

    def __init__(self, filename, model) -> None:
        self.filename = filename
        self.model = load_model(model)
        self.analysis = []

    def getResult(self) -> list:
        test_dir = os.path.join( 'spec' , '0~6000', self.filename )
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
                                    test_dir, # 目標目錄
                                    target_size = self.resize_shape, 
                                    batch_size = self.batch_size,
                                    class_mode = 'categorical' , 
                                    classes = self.my_class , 
                                    shuffle = False)
        result  = self.model.predict_generator(test_generator)
        np.set_printoptions(suppress=True)
        time = 0
        for index in range( len( test_generator ) ):
            real_labels = test_generator[index][1]
            for real in real_labels:
                predict = result[time]
                for i in predict:
                    # print ( '{:.2f}%'.format(i * 100 ) )
                    pass
                real_class = self.my_class[ np.argmax(real) ]
                predict_class = self.my_class[ np.argmax(predict) ]
                self.analysis.append(predict_class)

        return self.analysis

    # def getResult(self) -> list:
    #     return self.analysis


# =============================================================================
#    Smart Factory Service
# =============================================================================

class SmartFactoryService():
    def __init__(self, filename='', device='', model='', queue=None) -> None:
        # Thread.__init__(self)
        self.filename = filename
        self.device = device
        self.model = model
        self.queue = queue

    # override
    def run(self) -> None:
        au = Audio(self.filename, device=self.device)
        # print(au.getDeviceName())
        au.record()
        au.A_weighting()
        au.save_wav()
        Specgram(self.filename).toSpecgram()
        self.result = AI_analysis(self.filename, self.model).getResult()
        self.queue.put(self.result)
        print(self.result)


# if __name__ == '__main__':
#     filename = 'demo'
#     model_file = './model.h5'
#     queue = Queue()
#     sfs = SmartFactoryService(filename, device=Audio.DEVICE_1, model=model_file, queue=queue)
#     sfs.run()
#     print('main', queue.get())
#
