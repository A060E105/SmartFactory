import numpy as np
from tensorflow.keras import layers , models , Model
from tensorflow.keras.layers import Input , Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as pltrom 

resize_shape = ( 100 , 200 ) #y,x
train_shape =  tuple( list(resize_shape) + [3] )
batch_size = 32

my_class = ['NG', 'OK']

def AI_analysis(filename):


    save_model_name = 'model.h5'
    model = load_model(save_model_name)
    # model.summary()

    test_dir = os.path.join( 'spec' , '0~6000', filename )

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
                                test_dir, # 目標目錄
                                target_size = resize_shape, 
                                batch_size = batch_size,
                                class_mode = 'categorical' , 
                                classes = my_class , 
                                shuffle = False)

    test_loss, test_acc = model.evaluate_generator(test_generator)
    # print('acc:'  ,test_acc)
    # print('loss:' ,test_loss)

    result  = model.predict_generator(test_generator)

    np.set_printoptions(suppress=True)

    time = 0
    analysis = []
    for index in range( len( test_generator ) ):
        real_labels = test_generator[index][1]
        for real in real_labels:
            predict = result[time]
            for i in predict:
                # print ( '{:.2f}%'.format(i * 100 ) )
                pass
            real_class = my_class[ np.argmax(real) ]
            predict_class = my_class[ np.argmax(predict) ]
            analysis.append(predict_class)
        # print (f"np.argmax(real): {np.argmax(real)}")
        # print (f"real class: {real_class}")
        # print (f"np.argmax(predict): {np.argmax(predict_class)}")
        # print (f"predict class: {predict_class}")
    # return predict_class
    return analysis