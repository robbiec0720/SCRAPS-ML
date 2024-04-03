from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers 
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import keras
import PIL
import cv2
import json


font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2



def load_model(filename:str):
    input_shape = (256, 256, 3)

    bmodel = InceptionV3(weights="imagenet", input_shape=input_shape, include_top=False)
    bmodel.trainable = False
    model = Sequential()
    model.add(bmodel)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(36, activation='softmax'))
    model.load_weights(filename)
    return model

def capture():    
    model = keras.models.load_model('models/model.keras')
    classes = []
    with open("output.json", "r") as f:
        classes = json.load(f)
    

    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        nf = cv2.resize(frame, (256, 256))
        nf_tensor = tf.convert_to_tensor(nf)
        nf_tensor = tf.expand_dims(nf, 0)
        pred = model.predict(nf_tensor)
        max_indx = 0
        max_val = 0
        for i, val in enumerate(pred[0]):
            if val > max_val:
                max_val = val
                max_indx = i
        # print(classes[max_indx])
        
        frame = cv2.putText(frame, classes[max_indx], org, font, fontScale, color, thickness, cv2.LINE_AA )
        cv2.imshow('frame', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid.release()
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    # model = load_model('models/temp_model.hdf5')
    # save_model(model)
    capture()
    pass

