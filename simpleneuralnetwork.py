from numpy import exp, array, random, dot
inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synapticWeights = 2 * random.random((3, 1)) - 1 #set random weight
for iteration in range(10000):
	#calculate the neuron output
	output = 1 / (1 + exp(-(dot(inputs, synapticWeights))))
	#Adjust the wight
	synapticWeights += dot(inputs.T, (outputs - output) * output * (1 - output))
#Reduce the synapticWeights to 1x1 array and normalize for final answer!
print(1 / (1 + exp(-(dot(array([1, 0, 0]), synapticWeights)))))
