import numpy as np
import matplotlib.pyplot as plt
import Reservoir_ESN_1_0
from numpy import linalg

def recti():
    value = np.linspace(0, np.pi, steps).reshape(steps, 1)
    for v in range(len(value)):
        if np.pi / 4 <= value[v] <= 3 * np.pi / 4:
            value[v] = 1
        else:
            value[v] = 0
    return value


test_neurons = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 1000])
test_connect = np.array([2, 4, 7, 10, 20, 50])

for c in test_connect:
    test_error = np.zeros(test_neurons.shape)
    counter = 0
    print(c)
    for o in test_neurons:
        np.random.seed(20)
        steps = 100
        Neurons = o
        interval_count = 100
        connectivity = c/o
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

        # training_data_inp = np.concatenate((Interval_complete, x), axis=1)
        training_data_inp = x

        Reservoir1 = Reservoir_ESN_1_0.Reservoir(Neurons, training_data_inp.shape[1], 1, connectivity, feedback)
        ReservoirComp = Reservoir_ESN_1_0.ESN_Reservoir(Reservoir1, interval_count * steps)

        ReservoirComp.train(training_data_inp.T, training_data_out.T)
        """
        plt.plot(Interval_complete, x)
        plt.plot(Interval_complete, training_data_out)
        plt.plot(Interval_complete, ReservoirComp.Result.T)
        plt.legend(["Sine-Rect-Pulse", "Correct Result", "Reservoir Training Result"])
        plt.show()
        """
        # training_error = ((training_data_out - ReservoirComp.Result.T)**2).mean()
        # print('\r{}'.format(training_error))

        # guessing_data_inp = np.concatenate((Interval_complete + interval_count * np.pi + ReservoirComp.time_per_step,
        # guess_inp), axis=1)
        guessing_data_inp = guess_inp
        guessed_Result = ReservoirComp.guess(guessing_data_inp.T)

        test_error[counter] = ((correct_guess - guessed_Result.T)**2).mean()
        # print('\r{}'.format(test_error))

        """
        plt.plot(Interval_complete, guessing_data_inp[:, 0])
        plt.plot(Interval_complete, correct_guess)
        plt.plot(Interval_complete, guessed_Result.T)
        plt.legend(["Sine-Rect-Pulse", "Correct Result", "Reservoir Result"])
        plt.show()
        """
        counter += 1
    plt.plot(test_neurons, test_error)


plt.legend(["{} Connected Neurons".format(test_connect[0]),
            "{} Connected Neurons".format(test_connect[1]),
            "{} Connected Neurons".format(test_connect[2]),
            "{} Connected Neurons".format(test_connect[3]),
            "{} Connected Neurons".format(test_connect[4]),
            "{} Connected Neurons".format(test_connect[5])])
plt.title("Error with different Connectivity")
plt.xlabel("Number of Neurons")
plt.ylabel("Mean Square Error")
plt.yscale("log")
plt.show()
