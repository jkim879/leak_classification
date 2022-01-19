#누수음 분류 모델을 위한 패키지들을 불러오기-----------------------------
import sys
import wave
import librosa
from librosa import display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
from python_speech_features import mfcc
from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
#------------------------------------------------------------------------
dir = os.path.dirname(os.path.abspath(__file__))

#wav 파일의 전체 오디오 길이를 가져오기.--------------------------------------
def getDuration(filename):
    
    audio = wave.open(filename)
    frames = audio.getnframes()
    rate = audio.getframerate()
    duration = frames / float(rate)
    if duration >= 5 and duration < 6 :
        duration = 5
        
    elif duration >= 3 and duration < 4:
        duration = 3 

    elif duration >= 2 and duration < 3:
        duration = 2

    elif duration >= 4 and duration < 5:
        duration = 4 
    
    else:
        duration = 10 

    return duration
#------------------------------------------------------------------------


#1초 깨끗한 이미지를 만들기 위한 깨끗한 1초 알고리즘 함수 적용-------------------------------------

def getValue(filename):
    
    
    SEC_0_1 = 2205                #0.1초의 데이터 개수
    SEC_1  = 22050                #1초의 데이터 개수

    duration = getDuration(filename)

    s_fft = [] #FFT를 위한 리스트
    data,samplerate  = librosa.load(filename, sr = SEC_1, duration= duration) #오디오 파일을 리브로사를 이용하여 로드, 1.파일이름 2. samplerate = 22050, duration = 앱에서 세팅한 값

    i_time = (duration - 1) * 10 - 1 #5초인 경우, (5-1) * 10 - 1 = 39, #3초인 경우, (3-1) * 10 -1 = 19 등 

    
    for i in range(i_time): #i_time의 값만큼 반복 
        u_data = data[(i+1)*SEC_0_1:(i+1)*SEC_0_1+SEC_1] #u_data = data를 1초씩 자른 데이터
        s_fft.append(np.std(u_data)) #s_fft의 비어있는 리스트에 u_data의 표준편차를 추가 
    

    a=np.argmin(s_fft)+ 1 #s_fft 리스트중의 최소값의 인덱스
    #a = np.argmin(s_fft)


    tfa_data = data[a*SEC_0_1:a*SEC_0_1+SEC_1] #최소 표준편차를 가지는 인덱스의 1초데이터만 fft변환 후 변환 값의 절대값

    return tfa_data

#-----------------------------------------------------------------------------


#누수음 스펙트로그램 이미지를 만든 뒤 저장.---------------------------------------------------
def drawSpec(filename):
    
    signal = getValue(filename)

    #mfcc_feature = mfcc(signal) 

    M = librosa.feature.melspectrogram(signal, sr = 22050, fmin = 50, fmax = 2800, #max hz
                                        n_fft= 2048, #음성의 길이를 얼마만큼 자를 것인지
                                        hop_length=512,  #얼마만큼 시간주기를 이동하면서 분석할것인지
                                        n_mels = 96,  #칼라맵의 해상도
                                        power = 2) #default value 

    log_power = librosa.power_to_db(M, ref = np.max) #log scale로 변환
        

    # plotting the leak/noleak spectrogram and save as png file (just the image)
    pylab.figure(figsize=(2.16,2.16))
    pylab.axis('off') #no x or y axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(log_power, cmap=cm.jet)
    pylab.savefig(dir + '/1secmfcc_image.png', bbox_inches=None, pad_inches=0) #save imamge

#------------------------------------------------------------------------------------------------



#학습을 통해 만들어진 가중치 모델을 불러와 해당 누수음 스펙트로그램 이미지가 누수인지 아닌지를 판단-------
def getClasses(filename):
    #load model

    #dir = os.getcwd()
    #dir = '/home/nelow/backend/'
    # dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model(dir + '/220118_binary_classification.h5')

    drawSpec(filename)

    path = dir + "/1secmfcc_image.png"
    img = image.load_img(path , target_size=(150, 150))

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0) #차원 증가
 
    images = np.vstack([x])


    classes = model(images)
    
    return classes 

#----------------------------------------------------------------------------------------------------





#누수인지 아닌지를 어떻게 표출할지를 결정---------------------------------------------------------
def leakornot(filename):
    
    leak = "LEAK"
    noleak = "NO LEAK"
    
    if filename is None:
        print("file load failed")
        sys.exit()
        
    classes = getClasses(filename)

    
    if classes[0][0] == 1 and classes[0][1] == 0:
        print(leak)

    elif classes[0][0] == 0 and classes[0][1] == 1:
        print(noleak)

    else:
        print(noleak)

#-----------------------------------------------------------------------------------------------------


#최종 실행 코드--------------------------------------------------------------------------------------
def main():

    filename = sys.argv[1]

    #getDuration(filename)

    #drawSpec(filename)

    #getClasses(filename)

    leakornot(filename)

if __name__ == "__main__":
    main()  

  
#-------------------------------------------------------------------------------------------------
