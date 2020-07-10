import numpy as np
import matplotlib.pyplot as plt
import Reservoir_ESN_1_0


def recti():
    value = np.linspace(0, np.pi, steps).reshape(steps, 1)
    for v in range(len(value)):
        if np.pi / 4 <= value[v] <= 3 * np.pi / 4:
            value[v] = 1
        else:
            value[v] = 0
    return value


steps = 200
Neurons = 2000
interval_count = 40
connectivity = 0.02
feedback = 0

pi_Interval = np.linspace(0, np.pi, steps).reshape(steps, 1)
Interval_complete = np.linspace(0, interval_count * np.pi, interval_count * steps).reshape(interval_count * steps, 1)
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

training_data_inp = np.concatenate((Interval_complete, x), axis=1)

Reservoir1 = Reservoir_ESN_1_0.Reservoir(Neurons, 2, 1, connectivity, feedback)
ReservoirComp = Reservoir_ESN_1_0.ESN_Reservoir(Reservoir1, interval_count * steps)

ReservoirComp.train(training_data_inp.T, training_data_out.T)

plt.plot(Interval_complete, x)
plt.plot(Interval_complete, training_data_out)
plt.plot(Interval_complete, ReservoirComp.Result.T)
plt.legend(["Sine-Rect-Pulse", "Correct Result", "Trained Result"])
plt.show()

guessing_data_inp = np.concatenate(
    (Interval_complete + interval_count * np.pi + ReservoirComp.time_per_step, guess_inp), axis=1)
guessed_Result = ReservoirComp.guess(guessing_data_inp.T)

plt.plot(guessing_data_inp[:, 0], guessing_data_inp[:, 1])
plt.plot(guessing_data_inp[:, 0], correct_guess)
plt.plot(guessing_data_inp[:, 0], guessed_Result.T)
plt.legend(["Sine-Rect-Pulse", "Correct Result", "Reservoir Result"])
plt.show()