import tensorflow as tf
import os
import numpy as np

from matplotlib import pyplot as plt
if not os.path.isdir('model'):
    os.mkdir('model')
    
#print('tensorflow version :',tf.__version__)
#print('is gpu aviliable?', tf.config.list_physical_devices())

def get_classes(x,y):
    indices_0 , _ = np.where(y == 0)
    indices_1 , _ = np.where(y == 1)
    indices_2 , _ = np.where(y == 2)
    indices_3 , _ = np.where(y == 3)
    indices_4 , _ = np.where(y == 4)
    indices_5 , _ = np.where(y == 5)
    indices_6 , _ = np.where(y == 6)
    indices_7 , _ = np.where(y == 7)
    indices_8 , _ = np.where(y == 8)
    indices_9 , _ = np.where(y == 9)
        
    indices = np.concatenate([indices_0,indices_1,indices_2,indices_3,indices_4,indices_5,
                              indices_6,indices_7,indices_8,indices_9], axis=0)    

    x = x[indices]
    y = y[indices]
    
    count = x.shape[0]
    indices = np.random.choice(range(count), count, replace = False)
    
    x = x[indices]
    y = y[indices]
    
    y = tf.keras.utils.to_categorical(y)
    
    return(x,y)

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train, y_train = get_classes(x_train, y_train)
x_test, y_test = get_classes(x_test, y_test)

#print(x_train.shape,y_train.shape)
#print(x_test.shape,y_test.shape)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_random_examples(x, y, p):
    indices = np.random.choice(range(x.shape[0]), 10, replace = False)
    
    x = x[indices]
    y = y[indices]
    p = p[indices]
    
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, 1+i)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
        plt.xlabel(class_names[np.argmax(p[i])], color = col)
    plt.show()
    
show_random_examples(x_train, y_train, y_train)
show_random_examples(x_test, y_test, y_test)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Dense, Flatten, Input

def Create_model():
    def add_convo_block(model, num_filters):
        model.add(Conv2D(num_filters, 3, activation = 'relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))
        return model
    model = tf.keras.Sequential()
    model.add(Input(shape = (32, 32, 3)))
    
    moddel = add_convo_block(model, 32)
    moddel = add_convo_block(model, 64)
    moddel = add_convo_block(model, 128)
    
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'] )
    return model
model = Create_model()
model.summary()

h = model.fit(
    x_train/255., y_train,
    validation_data=(x_test/255., y_test),
    epochs=50, batch_size=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2),
        tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5', save_best_only=True,
                                          save_weights_only=False, monitor='val_accuracy')
    ]
)


losses = h.history['loss']
accs = h.history['accuracy']
val_losses = h.history['val_loss']
val_accs = h.history['val_accuracy']
epochs = len(losses)

plt.figure(figsize=(12, 4))
for i, metrics in enumerate(zip([losses, accs], [val_losses, val_accs], ['Loss', 'Accuracy'])):
    plt.subplot(1, 2, i + 1)
    plt.plot(range(epochs), metrics[0], label='Training {}'.format(metrics[2]))
    plt.plot(range(epochs), metrics[1], label='Validation {}'.format(metrics[2]))
    plt.legend()
plt.show()

model = tf.keras.models.load_model('models/model_0.913.h5')
preds = model.predict(x_test/255.)

show_random_examples(x_test, y_test, preds)
