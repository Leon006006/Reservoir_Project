import numpy as np
import matplotlib.pyplot as plt
import Reservoir_ESN_1_0

steps = 3000
Neurons = 800
End_time_training = 40
End_time_guess = 50
connectivity = 0.01
feedback_test = np.arange(0, 5).reshape(5, 1)
feedback_vec = np.zeros((feedback_test.shape[0], 1))
error = np.zeros((feedback_test.shape[0], 10))
j = 0
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('Error')
ax1.set_xlabel('Feedback weight')

while j < 10:
    for i in feedback_test:
        print(j, ",", i)

        current_feedback = 0.05 * (3+feedback_test[i, 0])
        feedback_vec[i, :] = current_feedback

        x = np.linspace(0, End_time_training, num=steps).reshape(steps, 1)
        y = np.sin(x)

        Reservoir1 = Reservoir_ESN_1_0.Reservoir(Neurons, 1, 1, connectivity, current_feedback)
        ReservoirComp = Reservoir_ESN_1_0.ESN_Reservoir(Reservoir1, steps)

        ReservoirComp.train(x.T, y.T)

        Start_time = End_time_training + ReservoirComp.time_per_step
        time_Vec = np.arange(Start_time, End_time_guess, ReservoirComp.time_per_step)
        time_Vec = time_Vec.reshape(1, time_Vec.shape[0])

        guessed_Result = ReservoirComp.guess(time_Vec)
        error[i, j] = np.linalg.norm(guessed_Result - y)

    plt.plot(feedback_vec, error[:, j])
    j = j + 1
plt.show()


plt.plot(x, y)
plt.plot(x, ReservoirComp.Result.T)
plt.plot(time_Vec.T, np.sin(time_Vec).T)
plt.plot(time_Vec.T, guessed_Result.T)
plt.legend(["Training exact function", "Training trained result", "Predicted exact", "Predicted Reservoir"])
plt.show()