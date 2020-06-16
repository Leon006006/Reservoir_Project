import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import Reservoir_ESN_1_0

from scipy import linalg
from scipy.sparse import random

def recti():
	value = np.linspace(0, np.pi, steps).reshape(steps, 1)
	for v in range(len(value)):
		if value[v] >= np.pi/4 and value[v] <= 3*np.pi/4:
			value[v] = 1
		else:
			value[v] = 0
	return value

steps = 200
Neurons = 1000
intervall_count = 30
connectivity = 0.1

pi_intervall = np.linspace(0, np.pi, steps).reshape(steps, 1)
Intervall_complete = np.linspace(0, intervall_count * np.pi, intervall_count*steps).reshape(intervall_count*steps, 1)
sin_vec = np.sin(pi_intervall)
rect_vec = recti()

guess_inp = rect_vec
x = sin_vec
training_data_out = np.full((steps, 1), 0)
correct_guess = np.full((steps, 1), 1)
for intervall in range(0, intervall_count-1):
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

training_data_inp = np.concatenate((Intervall_complete, x), axis=1)

Reservoir1 = Reservoir_ESN_1_0.Reservoir(Neurons, 2, 1, connectivity, 0)
ReservoirComp = Reservoir_ESN_1_0.ESN_Reservoir(Reservoir1, intervall_count * steps)

ReservoirComp.train(training_data_inp.T, training_data_out.T)

plt.plot(Intervall_complete, x)
plt.plot(Intervall_complete, training_data_out)
plt.plot(Intervall_complete, ReservoirComp.Result.T)
plt.legend(["Sinus-Rect-Impuls", "Exaktes Ergebnis", "Trainiertes Ergebnis"])
plt.show()

guessing_data_inp = np.concatenate((Intervall_complete+intervall_count*np.pi+ReservoirComp.time_per_step, guess_inp), axis=1)
guessed_Result = ReservoirComp.guess(guessing_data_inp.T)

plt.plot(guessing_data_inp[:, 0], guessing_data_inp[:, 1])
plt.plot(guessing_data_inp[:, 0], guessed_Result.T)
plt.plot(guessing_data_inp[:, 0], correct_guess)
plt.legend(["Sinus-Rect-Impuls", "Ergebnis Reservoir", "Richtiges Ergebnis"])
plt.show()
