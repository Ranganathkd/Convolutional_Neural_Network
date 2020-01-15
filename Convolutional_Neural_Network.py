from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Flatten, Embedding
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)

np.unique(y_train)

x_train = x_train/np.max(x_train)
x_test = x_test/np.max(x_test)

model = build_model()
print(model.summary())

hist = model.fit(x_train, y_train, epochs=20, batch_size=18, validation_split=0.1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['training', 'Validation'], loc='best')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['training', 'Validation'], loc='best')
plt.show()