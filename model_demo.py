from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers 
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import cv2

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
    model.loadweights(filename)
    return model


if __name__ == '__main__':
    # model = load_model('models/temp_model.hdf5')
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid.release()
    cv2.destroyAllWindows()
    pass


