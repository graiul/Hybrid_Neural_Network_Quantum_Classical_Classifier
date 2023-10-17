import numpy as np
import tensorflow as tf #Keras
from qiskit import QuantumCircuit, execute, transpile, assemble
# from qiskit import IBMQ
from qiskit_ibm_provider import IBMProvider
import qiskit_aer as Aer
# from scipy.optimize import minimize
import matplotlib.pyplot as plt
'''
Aici este un exemplu in care reteaua neuronala are mai mult
de un strat.
Este folosit si calculul cuantic pentru a antrena reteaua respectiva.

Mai am de rezolvat:
* WARNING:tensorflow:You are casting an input of type complex128 to an incompatible dtype float64.  This will discard the imaginary part and may not be what you intended.
* Pentru plotare, nu mai trebuie calculate din nou predictiile cuantice.
* Pentru micsorarea timpului de rulare, voi crea circuite care vor avea 
encodate mai multe puncte.
'''

# def authenticate_ibm_quantum(token, url='https://auth.quantum-computing.ibm.com/api'):
#     try:
#         IBMQ.save_account(token, overwrite=True, url=url)
#         provider = IBMQ.load_account()
#         print("Authentication successful!")
#         return provider
#     except Exception as e:
#         print("Authentication failed:", e)
#         return None
#
# # Provide your IBM Quantum token here
# my_ibm_token = 'YOUR_IBMQ_TOKEN'
#
# # Call the authentication function
# provider = authenticate_ibm_quantum(my_ibm_token)



# Simulate the quantum circuit using Qiskit
# Corrected version of simulate_quantum_circuit
# def simulate_quantum_circuit(data_point, quantum_circuit_params):
#     circuit = QuantumCircuit(1)
#     circuit.rx(data_point[0] * quantum_circuit_params[0], 0)
#     circuit.ry(data_point[1] * quantum_circuit_params[1], 0)
#
#     # IBMQ.save_account('', overwrite=True)
#     # IBMQ.update_account()
#     # provider = IBMQ.load_account()
#
#     # backend = Aer.get_backend('statevector_simulator')
#     backend = Aer.StatevectorSimulator()
#
#     # backend = provider.get_backend('simulator_statevector')
#
#     # job = assemble(transpile(circuit, backend=backend), shots=1)
#     # result = backend.run(job).result()
#
#     result = backend.run(circuit).result()
#
#
#     quantum_predictions = np.real(result.get_statevector(circuit))
#     return quantum_predictions

# Define the Quantum Neural Network
class QuantumNN:

    def __init__(self, live_flag):
        self.live_flag = live_flag
        if self.live_flag == 'y':
            API_access_token = input('Insert API access token: ')
            self.provider = IBMProvider(token=API_access_token)
            # IBMProvider.save_account(token='e6bbad0f51ab787aec48bed242c422777f1680f0428e19b34e19bbcd467a2faff2bdcedd0cbb04b19f4bb700f66900728f778d4bc8226785e6c35dc818374ac8')
            # provider = IBMProvider()

    def forward(self, data_point, quantum_circuit_params):
        circuit = QuantumCircuit(1)
        circuit.rx(data_point[0] * quantum_circuit_params[0], 0)
        circuit.ry(data_point[1] * quantum_circuit_params[1], 0)

        backend = None
        if self.live_flag == 'y':
            circuit.measure_all()

            backend = self.provider.get_backend('simulator_statevector')
            job = execute(circuit, backend, shots=8192)
            result = job.result()
            counts = result.get_counts(circuit)
            print('Counts:',counts)
            total_counts = sum(counts.values())
            exp_x = (counts.get('0', 0) - counts.get('1', 0)) / total_counts
            exp_y = (counts.get('0', 0) * 1j - counts.get('1', 0) * 1j) / total_counts
            quantum_predictions = np.array([[exp_x], [exp_y]], dtype=np.complex128)
            print('Quantum predictions:\n', quantum_predictions)
            return quantum_predictions

        elif self.live_flag == 'n':
            # backend = Aer.get_backend('statevector_simulator')
            backend = Aer.StatevectorSimulator()
            # result = execute(circuit, backend).result()
            result = backend.run(circuit).result()
            quantum_predictions = np.real(result.get_statevector(circuit))
            # print('Quantum predictions:', quantum_predictions)
            return quantum_predictions


# Define the classical neural network architecture using TensorFlow
class ClassicalNN(tf.Module):
    def __init__(self):
        self.dense1 = tf.Variable(tf.random.normal([2, 1], dtype=tf.float64))
        self.dense2 = tf.Variable(tf.random.normal([1, 1], dtype=tf.float64))
        # print(self.dense2)
    def __call__(self, inputs, classical_params):
        # print('Inputs:',inputs)
        # print(type(inputs))
        # print(inputs.shape)
        inputs2D = inputs.reshape((1,2))
        # print('Inputs2D:',inputs2D)
        # print(inputs2D.shape)
        # print('self.dense1:',self.dense1)
        # print(type(self.dense1))
        # exit(0)
        inputs_double = tf.cast(inputs2D, dtype=tf.float64)
        x = tf.nn.sigmoid(tf.matmul(inputs_double, self.dense1))
        # print('x=',x)
        output = tf.nn.sigmoid(tf.matmul(self.dense2, x) + classical_params)
        # return tf.nn.sigmoid(tf.matmul(x, self.dense2)) * 2.0 - 1.0 + classical_params
        return output

# Define the hybrid cost function
# Corrected version of hybrid_cost method
# def hybrid_cost(params, data, labels):
#     quantum_params = params[:2]
#     classical_params = params[2:]
#
#     quantum_predictions = [simulate_quantum_circuit(x, quantum_params) for x in data]
#     print('quantum_predictions: ',quantum_predictions)
#     # for quantumPredictionsTensor in quantum_predictions:
#
#     classical_predictions = [classical_nn(x, classical_params) for x in data]
#     # print('classical_predictions: ',classical_predictions)
#     for cp in classical_predictions:
#         print(cp)
#
#     loss = tf.reduce_mean(tf.square(classical_predictions - quantum_predictions - labels))
#     return loss

# def hybrid_cost(self, quantum_predictions, classical_predictions, labels):
#     total_loss = 0.0
#
#     for quantum_pred, classical_pred, label in zip(quantum_predictions, classical_predictions, labels):
#         # Convert quantum_pred and classical_pred to tensor with dtype=tf.float64
#         quantum_pred_tensor = tf.convert_to_tensor(quantum_pred, dtype=tf.float64)
#         classical_pred_tensor = tf.convert_to_tensor(classical_pred, dtype=tf.float64)
#
#         # Convert quantum_pred to shape (2,) and then reshape to (2, 1)
#         reshaped_quantum_pred = quantum_pred_tensor.numpy().reshape((2, 1))
#
#         # Create a classical neural network instance
#         classical_nn = ClassicalNN()
#
#         # Calculate the classical predictions
#         classical_predictions = classical_nn(reshaped_quantum_pred, self.classical_params)
#
#         # Calculate the loss for this data point
#         loss = tf.reduce_mean(tf.square(classical_predictions - label))
#
#         # Add the loss to the total
#         total_loss += loss
#
#     # Calculate the overall loss
#     overall_loss = total_loss / len(quantum_predictions)
#
#     return overall_loss

class HybridModel:
    def __init__(self):
        self.quantum_nn = QuantumNN(use_live_ibm_simulator_flag)
        self.classical_nn = ClassicalNN()
        self.classical_params = tf.Variable(0.0, dtype=tf.float64)

    def hybrid_cost(self, quantum_predictions, classical_predictions, labels):
        total_loss = 0.0
        for quantum_pred, classical_pred, label in zip(quantum_predictions, classical_predictions, labels):
            loss = tf.reduce_mean(tf.square(classical_pred - label))
            total_loss += loss
        overall_loss = total_loss / len(quantum_predictions)
        return overall_loss

use_live_ibm_simulator_flag = input('Use live IBM Statevector Simulator (y/n): ')

# Generate synthetic data for blue and orange points
np.random.seed(0)
num_samples = 100
data_blue = np.random.rand(num_samples // 2, 2) * 0.4
data_orange = 0.5 + np.random.rand(num_samples // 2, 2) * 0.4
data = np.vstack((data_blue, data_orange))
labels = np.vstack((np.zeros((num_samples // 2, 1)), np.ones((num_samples // 2, 1))))

# # Generate synthetic data points and labels
# data_points = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
# labels = [4.0, 5.0, 6.0]

# # Initialize parameters for optimization
# initial_params = np.random.rand(4)

# Instantiate the Hybrid Model
hybrid_model = HybridModel()

# # Instantiate the classical neural network
# classical_nn = ClassicalNN()

# quantum_predictions = simulate_quantum_circuit(data, )
# classical_predictions = []

# Define the grid or random search space for quantum_circuit_params
param_ranges = [(-np.pi, np.pi), (-np.pi, np.pi)]

# Perform grid search or random search
best_cost = float('inf')
best_params = None
num_trials = 10

for _ in range(num_trials):
    random_params = [np.random.uniform(range[0], range[1]) for range in param_ranges]

    # Generate quantum and classical predictions for each data point using random_params
    quantum_predictions = []
    classical_predictions = []

    for data_point in data: # In the case of the blue and orange points.
    # for data_point in data_points:

        quantum_pred = hybrid_model.quantum_nn.forward(data_point, random_params)
        quantum_predictions.append(quantum_pred)

        reshaped_quantum_pred = np.array(quantum_pred).reshape((2, 1))
        classical_pred = hybrid_model.classical_nn(reshaped_quantum_pred, hybrid_model.classical_params)
        classical_predictions.append(classical_pred)

    # Calculate the hybrid cost using the generated predictions
    cost = hybrid_model.hybrid_cost(quantum_predictions, classical_predictions, labels)

    # Update best_params if a better cost is found
    if cost < best_cost:
        best_cost = cost
        best_params = random_params

print("\nBest Hybrid Cost:", best_cost)
print("Best Quantum Circuit Params:", best_params)
print()




# # Perform hybrid optimization using classical optimizer
# # The purpose of this code is to optimize the parameters
# # of your hybrid model (both quantum and classical components)
# # by minimizing the hybrid_cost function. The optimizer will adjust the
# # parameters in such a way that the hybrid cost is minimized,
# # which ideally results in better predictions from your hybrid model.
# result = minimize(
#     hybrid_cost,
#     initial_params,
#     # args=(data, labels),
#     args=(quantum_predictions, classical_predictions, labels),
#     method='L-BFGS-B',
#     options={'disp': True, 'maxiter': 100}
# )
#
# # Get the optimal parameters
# optimal_params = result.x
#
# # Evaluate the trained hybrid model
# quantum_params = optimal_params[:2]
# classical_params = optimal_params[2:]



# # Another variant:
#
# # Generate synthetic data points and labels
# data_points = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
# labels = [4.0, 5.0, 6.0]
#
# # Initial guess for quantum_circuit_params
# initial_params = [0.0, 0.0]  # Replace with your initial guess
#
# # Use the minimize function to optimize quantum_circuit_params
# result = minimize(
#     cost_function,
#     initial_params,
#     args=(data_points, labels),
#     method='L-BFGS-B',
#     options={'disp': True, 'maxiter': 100}
# )
#
# best_quantum_circuit_params = result.x
# best_cost = result.fun
#
# print("Best Quantum Circuit Params:", best_quantum_circuit_params)
# print("Best Hybrid Cost:", best_cost)


# Create a grid for visualization
x_vals = np.linspace(0, 1, 50)
y_vals = np.linspace(0, 1, 50)
grid = np.array([(x, y) for x in x_vals for y in y_vals])

# Predict class probabilities for each point in the grid
probs = []
for point in grid:
    quantum_pred = hybrid_model.quantum_nn.forward(point, best_params)
    # exit(0)
    reshaped_quantum_pred = np.array(quantum_pred).reshape((2, 1))
    classical_pred = hybrid_model.classical_nn(reshaped_quantum_pred, hybrid_model.classical_params)
    probs.append(classical_pred)

# Plot the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.scatter(data_blue[:, 0], data_blue[:, 1], color='blue', label='Blue points')
plt.scatter(data_orange[:, 0], data_orange[:, 1], color='orange', label='Orange points')

probs = np.array(probs).reshape(len(x_vals), len(y_vals))
decision_boundary = np.where(probs > 0.5, 1, 0)
plt.contourf(x_vals, y_vals, decision_boundary, alpha=0.2, levels=1, colors=['blue', 'orange'])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hybrid Quantum-Classical Classifier')
plt.legend()

# 1. Generate the quantum and classical predictions for the new data point.
# 2. Feed the quantum and classical predictions through the classical neural network.
# 3. Use the output of the classical neural network to make a classification decision.

# Generate a new data point for classification
# new_data_point = [0.2, 0.3] # Blue point.
# new_data_point = [0.6, 0.7] # Orange point.
# new_data_point = [0.2, 0.8] # Point outside the two regions.
print('Coordinates of a new point:\n\tExample data: \n\t[0.2, 0.3] # Blue point.\n\t[0.6, 0.7] # Orange point.\n\t[0.2, 0.8] # Point outside the two regions.')
# new_data_point = [0.6, 0.7] # Orange point.
# new_data_point = [0.2, 0.8] # Point outside the two regions.')
new_data_point = [float(input('X = ')), float(input('Y = '))]

# Generate quantum and classical predictions for the new data point using best_params
new_quantum_pred = hybrid_model.quantum_nn.forward(new_data_point, best_params)
reshaped_new_quantum_pred = np.array(new_quantum_pred).reshape((2, 1))
new_classical_pred = hybrid_model.classical_nn(reshaped_new_quantum_pred, hybrid_model.classical_params)

# threshold = 0.6  # Adjust this threshold as needed
threshold = float(input('Insert threshold value (in testing 0.6 has been used): '))
# Classify the new data point based on the new_classical_pred
if new_classical_pred > threshold:
    classification = "Orange"
else:
    classification = "Blue"

print(f"The new data point {new_data_point} is classified as {classification}.")

# Add a scatter plot marker for the new data point
# Plasarea punctului pe figura este facuta in functie de coordonate,
# nu in functie de predictia rezultata din model.
plt.scatter(new_data_point[0], new_data_point[1], color='yellow', marker='*', s=200, label='New Point')
# # Include the classification of the new data point in the legend
# if classification == "Blue":
#     plt.legend(handles=[blue_patch, orange_patch, star_marker], labels=['Blue points', 'Orange points', 'New Blue Point'])
# else:
#     plt.legend(handles=[blue_patch, orange_patch, star_marker], labels=['Blue points', 'Orange points', 'New Orange Point'])


plt.show()

# # Create a grid for visualization, in the case of blue and orange points.
# x_vals = np.linspace(0, 1, 50)
# y_vals = np.linspace(0, 1, 50)
# grid = np.array([(x, y) for x in x_vals for y in y_vals])
# print('Grid:',grid)
#
# # Predict class probabilities for each point in the grid
# probs = []
# for point in grid:
#     quantum_pred = simulate_quantum_circuit(point, quantum_params)
#     classical_pred = classical_nn(point, classical_params)
#     probs.append((quantum_pred, classical_pred))
#
# # Plot the decision boundary and data points
# plt.figure(figsize=(8, 6))
# plt.scatter(data_blue[:, 0], data_blue[:, 1], color='blue', label='Blue points')
# plt.scatter(data_orange[:, 0], data_orange[:, 1], color='orange', label='Orange points')
#
# probs = np.array(probs).reshape(len(x_vals), len(y_vals), 2)
# decision_boundary = np.where(probs[:, :, 0] > 0.5, 1, 0)
# plt.contourf(x_vals, y_vals, decision_boundary, alpha=0.2, levels=1, colors=['blue', 'orange'])
#
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Hybrid Quantum-Classical Classifier')
# plt.legend()
# plt.show()

