import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import random

class Reservoir:
	"""Class for Reservoir itself"""

	def __init__(self, Number_Neurons, Input_dim, Output_dim, connectivity, feedback):
		self.Input_dim = Input_dim
		self.Output_dim = Output_dim
		self.Number_Neurons = Number_Neurons
		self.feedback = feedback
		self.W_input = random(Number_Neurons, Input_dim).A
		self.W_intern = random(Number_Neurons, Number_Neurons, density=connectivity).A
		eigenvalues = linalg.eigvals(self.W_intern)
		spectral_radius = eigenvalues.max()
		self.W_intern = self.W_intern/(1.05 * abs(spectral_radius))
		if feedback != 0:
			self.W_feedback = random(Number_Neurons, Output_dim, density=connectivity).A

class ESN_Reservoir(object):
	"""Class for training ReadOut and Computing results"""

	def __init__(self, Reservoir, steps):
		self.Reservoir = Reservoir
		self.steps = steps
		self.Neuron_State = np.zeros((self.Reservoir.Number_Neurons,self.steps))
		self.z = np.zeros((self.Reservoir.Number_Neurons + self.Reservoir.Input_dim,steps))
		self.ReadOut = np.zeros((self.Reservoir.Output_dim,self.Reservoir.Number_Neurons + self.Reservoir.Input_dim))
		self.Result = np.zeros((self.Reservoir.Output_dim,steps))

	def train(self, Input, desired_Output):
		self.time_per_step = Input[0, 1]-Input[0, 0]
		from_intern = np.dot(self.Reservoir.W_intern, self.Neuron_State[:, 0].reshape(self.Reservoir.Number_Neurons, 1))
		from_input = np.dot(self.Reservoir.W_input, Input[:, 0].reshape(Input.shape[0], 1))

		if self.Reservoir.feedback != 0:
			from_feedback = np.dot(self.Reservoir.W_feedback, desired_Output[:, 0].reshape(desired_Output.shape[0], 1))
			self.Neuron_State[:, 0] = np.tanh((1-self.Reservoir.feedback) * from_intern + from_input + self.Reservoir.feedback * from_feedback).reshape(self.Reservoir.Number_Neurons, )
		elif self.Reservoir.feedback == 0:
			self.Neuron_State[:, 0] = np.tanh(from_intern + from_input).reshape(self.Reservoir.Number_Neurons,)
		else:
			print("Feedback not set")
			return
		self.z[:, 0] = np.concatenate((Input[:, 0], self.Neuron_State[:, 0]), axis=0)

		for steppystep in range(1, self.steps):
			if self.Reservoir.feedback != 0:
				self.Neuron_State[:, steppystep] = np.tanh((1-self.Reservoir.feedback) * (np.dot(self.Reservoir.W_intern, self.Neuron_State[:, steppystep-1]) + np.dot(self.Reservoir.W_input, Input[:, steppystep])) + self.Reservoir.feedback * np.dot(self.Reservoir.W_feedback, desired_Output[:, steppystep-1]))
			elif self.Reservoir.feedback == 0:
				self.Neuron_State[:, steppystep] = np.tanh(np.dot(self.Reservoir.W_intern, self.Neuron_State[:, steppystep-1]) + np.dot(self.Reservoir.W_input, Input[:, steppystep]))
			else:
				print("Feedback not set")
				return
			self.z[:, steppystep] = np.concatenate((Input[:, steppystep], self.Neuron_State[:, steppystep]), axis=0)

			if steppystep % 10 == 0:
				print('\rTraining-Progress: {}%'.format(np.round((steppystep+1)/self.steps * 100.), 2), end='')

		self.ReadOut = (np.dot(desired_Output, linalg.pinv(self.z)))
		self.Result = np.dot(self.ReadOut, self.z)

	def guess(self, time_Vec):
		from_intern = np.dot(self.Reservoir.W_intern, self.Neuron_State[:, self.Neuron_State.shape[1]-1].reshape(self.Reservoir.Number_Neurons, 1))
		from_input = np.dot(self.Reservoir.W_input, time_Vec[:, 0]).reshape(self.Reservoir.Number_Neurons, 1)
		if self.Reservoir.feedback != 0:
			from_feedback = np.dot(self.Reservoir.W_feedback, self.Result[0, self.Result.shape[1]-1])
			current_neuron_state = np.tanh(np.add((1-self.Reservoir.feedback) * np.add(from_intern, from_input), self.Reservoir.feedback * from_feedback))
		elif self.Reservoir.feedback == 0:
			current_neuron_state = np.tanh(np.add(from_intern, from_input))
		else:
			print("Feedback not set")
			return
		z_temp = np.concatenate((time_Vec[:, 0].reshape(self.Reservoir.Input_dim, 1), current_neuron_state), axis=0)
		guessed_result = np.dot(self.ReadOut, z_temp).reshape(self.Reservoir.Output_dim, 1)

		for steppystep in range(1, time_Vec.shape[1]):
			from_intern = np.dot(self.Reservoir.W_intern, current_neuron_state)
			from_input = np.dot(self.Reservoir.W_input, time_Vec[:, steppystep].reshape(self.Reservoir.Input_dim, 1))
			if self.Reservoir.feedback != 0:
				from_feedback = np.dot(self.Reservoir.W_feedback, guessed_result[:, steppystep-1]).reshape(self.Reservoir.Number_Neurons, 1)
				current_neuron_state = np.tanh(np.add((1 - self.Reservoir.feedback) * np.add(from_intern, from_input), self.Reservoir.feedback * from_feedback))
			elif self.Reservoir.feedback == 0:
				current_neuron_state = np.tanh(np.add(from_intern, from_input))
			else:
				print("Feedback not set")
				return
			z_temp = np.concatenate((time_Vec[:, steppystep].reshape(self.Reservoir.Input_dim, 1), current_neuron_state), axis=0)
			guessed_result = np.concatenate((guessed_result, (np.dot(self.ReadOut, z_temp)).reshape(self.Reservoir.Output_dim, 1)), axis=1)

		return guessed_result