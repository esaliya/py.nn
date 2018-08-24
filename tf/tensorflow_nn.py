import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt



fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_lbls), (test_imgs, test_lbls) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure()
plt.imshow(train_imgs[0])
plt.colorbar()
plt.gca().grid(False)

train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_imgs[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_lbls[i]])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_imgs, train_lbls, epochs=5)

test_loss, test_acc = model.evaluate(test_imgs, test_lbls)
print('Test accuracy:', test_acc)

predictions = model.predict(test_imgs)
predictions[0]
np.argmax(predictions[0])
test_lbls[0]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_imgs[i], cmap=plt.cm.binary)
    predicted_lbl = np.argmax(predictions[i])
    true_lbl = test_lbls[i]
    if predicted_lbl == true_lbl:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_lbl],
                                class_names[true_lbl],
                                color=color))
plt.show()