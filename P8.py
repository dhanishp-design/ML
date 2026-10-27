import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

# Activation Functions
sigmoid = 1/(1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0,x)

# Plotting the activation functions

# Plot Sigmoid 

plt.plot(x, sigmoid)
plt.title("Sigmoid")
plt.show()

# Plot Tanh 

plt.plot(x, tanh)
plt.title("Tanh")
plt.show()

# Plot ReLU

plt.plot(x, relu)
plt.title("ReLU")
plt.show()
