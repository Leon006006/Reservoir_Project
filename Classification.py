import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Reservoir_ESN_1_0 as Res


def recti():
    value = np.linspace(0, np.pi, steps).reshape(steps, 1)
    for v in range(len(value)):
        if np.pi / 4 <= value[v] <= 3 * np.pi / 4:
            value[v] = 1
        else:
            value[v] = 0
    return value


np.random.seed(4121)
steps = 20
Neurons = 10
interval_count = 20
connectivity = 0.1
feedback = 0

pi_Interval = np.linspace(0, np.pi, steps).reshape(steps, 1)
Interval_complete = np.linspace(0, interval_count * np.pi, interval_count * steps).reshape(interval_count * steps, 1)
noise = np.random.normal(0, 0.005, pi_Interval.shape)
sin_vec = np.sin(pi_Interval).reshape(steps, )
rect_vec = recti().reshape(steps, )

Training_Data = np.full((steps, interval_count), 1, dtype=float)
test_data = np.full((steps, interval_count), 1, dtype=float)
training_data_tag = np.full((1, interval_count), 0, dtype=int)
correct_guess = np.full((1, interval_count), 0, dtype=int)

for interval in range(0, interval_count):
    rand_numb = np.random.rand(1, 1)
    if rand_numb < 0.5:
        Training_Data[:, interval] = sin_vec
        training_data_tag[0, interval] = 0
        test_data[:, interval] = rect_vec
        correct_guess[:, interval] = 1
    else:
        Training_Data[:, interval] = rect_vec
        training_data_tag[0, interval] = 1
        test_data[:, interval] = sin_vec
        correct_guess[0, interval] = 0


#plt.plot(Interval_complete, Training_Data.T.reshape(steps*interval_count, 1))
tags_plot = np.full((steps, interval_count), 1, dtype=float)
for interval in range(interval_count):
    tags_plot[:, interval] = training_data_tag[0, interval] * np.full((steps, ), 1)
plt.plot(Interval_complete, tags_plot.T.reshape(steps*interval_count, ))
plt.show()

Reservoir1 = Res.Reservoir(Neurons, Training_Data.shape[0], 1, connectivity, feedback)
ReservoirComp = Res.ESN_Reservoir(Reservoir1, interval_count)

ReservoirComp.train(Training_Data, training_data_tag)
for interval in range(interval_count):
    tags_plot[:, interval] = ReservoirComp.Result[0, interval] * np.full((steps, ), 1)
plt.plot(Interval_complete, tags_plot.T.reshape(steps*interval_count, ))
plt.show()

guessed_Result = ReservoirComp.guess(test_data)
for interval in range(interval_count):
    tags_plot[:, interval] = guessed_Result[0, interval] * np.full((steps, ), 1)
plt.plot(Interval_complete, tags_plot.T.reshape(steps*interval_count, ))
plt.show()
