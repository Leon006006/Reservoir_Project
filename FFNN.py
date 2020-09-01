import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Reservoir_ESN_1_0 as Res

from tensorflow import keras


def recti():
    value = np.linspace(0, np.pi, steps).reshape(steps, 1)
    for v in range(len(value)):
        if np.pi / 4 <= value[v] <= 3 * np.pi / 4:
            value[v] = 1
        else:
            value[v] = 0
    return value


np.random.seed(4121)
steps = 200
Neurons = 1000
interval_count = 20
connectivity = 0.01
feedback = 0

pi_Interval = np.linspace(0, np.pi, steps).reshape(steps, 1)
Interval_complete = np.linspace(0, interval_count * np.pi, interval_count * steps).reshape(interval_count * steps, 1)
noise = np.random.normal(0, 0.005, pi_Interval.shape)
sin_vec = np.sin(pi_Interval)
rect_vec = recti()

guess_inp = rect_vec
x = sin_vec
training_data_out = np.full((steps, 1), 0)
correct_guess = np.full((steps, 1), 1)
for interval in range(0, interval_count - 1):
    rand_numb = np.random.rand(1, 1)
    if rand_numb < 0.5:
        x = np.concatenate((x, sin_vec), axis=0)
        training_data_out = np.concatenate((training_data_out, np.full((steps, 1), 0)), axis=0)
        guess_inp = np.concatenate((guess_inp, rect_vec), axis=0)
        correct_guess = np.concatenate((correct_guess, np.full((steps, 1), 1)), axis=0)
    else:
        x = np.concatenate((x, rect_vec), axis=0)
        training_data_out = np.concatenate((training_data_out, np.full((steps, 1), 1)), axis=0)
        guess_inp = np.concatenate((guess_inp, sin_vec), axis=0)
        correct_guess = np.concatenate((correct_guess, np.full((steps, 1), 0)), axis=0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, 1)),
    keras.layers.Dense(Neurons, activation='sigmoid'),
    keras.layers.Dense(Neurons, activation='sigmoid'),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x.reshape(interval_count*steps, 1, 1), training_data_out, epochs=20)

test_loss, test_acc = model.evaluate(guess_inp.reshape(interval_count*steps, 1, 1), correct_guess, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(guess_inp.reshape(interval_count*steps, 1, 1))

test_class_FFNN = np.full((interval_count*steps, ), 2, dtype=int)
for point in range(interval_count*steps):
    test_class_FFNN[point, ] = np.argmax(predictions[point, :])
plt.plot(Interval_complete[:2000], test_class_FFNN[:2000])
# plt.plot(Interval_complete, correct_guess)
plt.plot(Interval_complete[:2000], guess_inp[:2000])
plt.legend(["Result FFNN Classification", "Input Signal"])
plt.show()
