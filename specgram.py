import os
import io
import cv2
import wave
import numpy as np
import pylab as plt
from PIL import Image
from tqdm import tqdm
from numpy.lib import stride_tricks
from matplotlib import pyplot as plt

picture_width , picture_height = 100 , 50 #圖片長寬比
# with_cut    = True
CutTimeDef  = 2 #以1s截斷檔案
SpaceNumDef = 1 #每次取時間間隔
#freq_split_list = [ [0,24000] , [24000,48000] , [0,48000] , [10000,30000] ]
freq_split_list = [ [0,6000] ]
# save_split_audio_file = False  #是否儲存音檔
source_path     = './source/'
audio_out_path  = './audio/'
spec_path       = './spec/'

def my_mkdir( path ):
    now_path = ''
    for i in path.split('\\'):
        now_path = os.path.join( now_path , i )
        try:
            os.mkdir(now_path)
        except:
            pass

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
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

def logscale_spec(spec, sr=96000, factor=20.):
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

def plotstft( freq_split_list , im_width , im_height ,samplerate, samples , binsize=2**10, plotpath=None, colormap="jet"):
    s = stft(samples, binsize)
    sshow_origin, freq = logscale_spec(s, factor=1.0, sr=samplerate)
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
        
        
        timebins, freqbins = np.shape(ims)
        #print("timebins: ", timebins)
        #print("freqbins: ", freqbins)
        # plt.imshow(np.transpose(ims), origin="None", aspect="auto", cmap="jet", extent = None, interpolation='None', vmin= -160, vmax= 0)
        plt.imshow(np.transpose(ims), aspect="auto", cmap="jet", extent = None, interpolation='None', vmin= -160, vmax= 0)
        plt.axis('off') 
        fig = plt.gcf()
        fig.set_size_inches(  im_width , im_height  ) #dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        # 去除空白code https://blog.csdn.net/qq_30708445/article/details/103190850?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-1
        img_arr = get_img_from_fig( fig , dpi = 1 )
        image_list.append( img_arr )
        plt.clf()
    return np.array(image_list)


def CutFile( file_path , with_cut_file = True ):
    f = wave.open( file_path , "rb")
    params = f.getparams()
    
    nchannels, sampwidth, framerate, nframes = params[:4]
    CutFrameNum = framerate* CutTimeDef
    # 讀取格式資訊
    # 一次性返回所有的WAV檔案的格式資訊，它返回的是一個組元(tuple)：聲道數, 量化位數（byte    單位）, 採
    # 樣頻率, 取樣點數, 壓縮型別, 壓縮型別的描述。wave模組只支援非壓縮的資料，因此可以忽略最後兩個資訊
    '''
    print("CutFrameNum :%d" % (CutFrameNum))
    print("nchannels   :%d" % (nchannels))
    print("sampwidth   :%d" % (sampwidth))
    print("framerate   :%d" % (framerate))
    print("nframes     :%d" % (nframes))
    '''
    str_data = f.readframes(nframes)
    f.close()# 將波形資料轉換成陣列
    # Cutnum =nframes/framerate/CutTimeDef
    # 需要根據聲道數和量化單位，將讀取的二進位制資料轉換為一個可以計算的陣列
    wave_data = np. frombuffer(str_data, dtype=np.short)
    
    if nchannels >= 2:  #轉換成雙聲道
        wave_data.shape = -1, 2
        wave_data = wave_data.T
    else:
        wave_data = wave_data.reshape( 1,wave_data.shape[0] )
    
    if with_cut_file:
        ptr_start = 0
        time = 0
        total_time = (wave_data.shape[1] / framerate / SpaceNumDef) - CutTimeDef #推測總共圖片數量
        total_time = int(total_time) + 1
        with tqdm(total = total_time) as pbar:
            while 1:
                ptr_end = ptr_start + CutTimeDef * framerate
                ptr_end = int(ptr_end)
                if ptr_end <= nframes:
                    temp_dataTemp = wave_data[0][ptr_start:ptr_end]  #分割音檔
                    image_list = plotstft( freq_split_list , picture_width , picture_height , framerate , temp_dataTemp ) #轉換成圖片
                    ptr_start += SpaceNumDef * framerate
                    ptr_start = int(ptr_start)
                    time += 1
                    pbar.update(1)
                    #print('\n' , ptr_start , ptr_end, '\n')
                    yield [sampwidth , framerate , temp_dataTemp] , image_list
                else:
                    break
    else:
        temp_dataTemp = wave_data[0]
        image_list = plotstft( freq_split_list , picture_width , picture_height , framerate , temp_dataTemp ) #轉換成圖片
        yield [sampwidth , framerate , temp_dataTemp] , image_list


def convert_to_specgram(filename, with_cut=True, save_split_audio_file=False):
    audio_path = f'{source_path}{filename}.wav'
    audio_name = filename.split('.')[0]
    print (audio_name)
    index = 0

    for audio_info , image_list in CutFile( audio_path , with_cut):
        index += 1
        if save_split_audio_file:
            sampwidth = audio_info[0]
            framerate = audio_info[1]
            temp_dataTemp = audio_info[2]
            file_name = '{}_{}.wav'.format(audio_name,index)
            audio_save_dir = os.path.join( audio_out_path , audio_name )
            my_mkdir(audio_save_dir)
            audio_save_path =  os.path.join( audio_save_dir , file_name )  #設定音檔儲存路徑
            f = wave.open( audio_save_path, "wb")
            f.setnchannels(1) #固定輸出單聲道
            f.setsampwidth(sampwidth)
            f.setframerate(framerate)
                # 將wav_data轉換為二進位制資料寫入檔案
            f.writeframes(temp_dataTemp.tobytes() )
            f.close()
        for freq_info,image in zip(freq_split_list , image_list):
            img_bin = Image.fromarray( image.astype( np.uint8 )  , 'RGB')
            freq_str = '{}~{}'.format( freq_info[0] , freq_info[1] )
            image_dir = os.path.join( spec_path , freq_str , audio_name, 'OK' )
            file_name = '{}_{}.png'.format( audio_name.split('.')[0] , index )
            my_mkdir(image_dir)
            image_path = os.path.join( image_dir , file_name)  #設定圖片儲存路徑
            img_bin.save( image_path )


if __name__ == '__main__':
    filename = 'test.wav'
    convert_to_specgram(filename)
