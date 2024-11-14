import numpy as np
import pickle
from mnist import Mnist
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

# Load MNIST dataset
mnist = Mnist()
(x_train, y_train), (x_test, y_test) = mnist.load()
# Initialize the neural network
network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

# Hyperparameters
iterations = 100000
batch_size = 16
learning_rate = 0.01
train_size = x_train.shape[0]
print(train_size)
iter_per_epoch = int(max(train_size // batch_size, 1))


train_losses = []
train_accs = []
test_accs = []

# Training loop
for i in range(iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)

    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learning_rate*grads[key]

    ## this is for plotting losses over time
    train_losses.append(network.loss(x_batch, y_batch))

    if i%iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        train_accs.append(train_acc)
        test_acc = network.accuracy(x_test, y_test)
        test_accs.append(test_acc)
        print(f'train acc, test_acc : {train_acc}, {test_acc}')

# Save the entire trained model to a pickle file
model_filename = "raokhande_mnist_entire_model.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(network, f)
network.update_layers()
print(f"Entire model saved as {model_filename}")

