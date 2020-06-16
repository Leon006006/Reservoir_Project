import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import Reservoir_ESN_1_0

from scipy import linalg
from scipy.sparse import random

steps = 10000
Neurons = 1000
End_time_training = 50
connectivity = 0.01
feedback = 1

x = np.linspace(0, End_time_training, num=steps).reshape(steps, 1)
y = np.sin(x)

Reservoir1 = Reservoir_ESN_1_0.Reservoir(Neurons, 1, 1, connectivity, feedback)
ReservoirComp = Reservoir_ESN_1_0.ESN_Reservoir(Reservoir1, steps)

ReservoirComp.train(x.T,y.T)

plt.plot(x,y)
plt.plot(x,ReservoirComp.Result.T)

Start_time 	= End_time_training + ReservoirComp.time_per_step
End_time 	= 75
time_Vec 	= np.arange(Start_time,End_time,ReservoirComp.time_per_step)
time_Vec	= time_Vec.reshape(1,time_Vec.shape[0])

guessed_Result	= ReservoirComp.guess(time_Vec)

plt.plot(time_Vec.T,np.sin(time_Vec).T)
plt.plot(time_Vec.T,guessed_Result.T)
plt.legend(["Training Exakt", "Training Gelernt", "Vorhergesagt Exakt", "Vorhergesagt Reservoir"])
plt.show()