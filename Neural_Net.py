from sklearn.datasets import fetch_openml
import numpy

L = 3
n = [784, 128, 64, 10]

def he_init(out_dim, in_dim):
    return numpy.random.randn(out_dim, in_dim) * (numpy.sqrt(2.0 / in_dim))

def xavier_init(out_dim, in_dim):
    return numpy.random.randn(out_dim, in_dim) * (numpy.sqrt(1.0 / in_dim))

Weights1 = he_init(n[1], n[0])
Weights2 = he_init(n[2], n[1])
Weights3 = xavier_init(n[3], n[2])
Bias1 = numpy.zeros((n[1], 1))
Bias2 = numpy.zeros((n[2], 1))
Bias3 = numpy.zeros((n[3], 1))

def relu(z):
    return numpy.maximum(0, z)

def reluDerivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z_shift = z - numpy.max(z, axis=0, keepdims=True)
    expZ = numpy.exp(z_shift)
    return expZ / numpy.sum(expZ, axis=0, keepdims=True)

def prepareData():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    x = mnist.data.astype(numpy.float64) / 255.0
    y = mnist.target.astype(int)

    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    x_train = x_train.T
    x_test = x_test.T

    def one_hot(y, num_classes=10):
        m = y.shape[0]
        Y = numpy.zeros((num_classes, m))
        for i, label in enumerate(y):
            Y[label, i] = 1.0
        return Y
    
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    return x_train, y_train, x_test, y_test

def FeedForward(Activation0):

    #Layer 1 calculations

    OutputLayer1 = Weights1 @ Activation0 + Bias1
    Activation1 = relu(OutputLayer1)

    #Layer 2 calculations

    OutputLayer2 = Weights2 @ Activation1 + Bias2
    Activation2 = relu(OutputLayer2)

    #Layer 3 calculations

    OutputLayer3 = Weights3 @ Activation2 + Bias3
    Activation3 = softmax(OutputLayer3)

    #Final data

    cache = {
        "A0" : Activation0,
        "A1" : Activation1,
        "A2" : Activation2,
        "Z1" : OutputLayer1,
        "Z2" : OutputLayer2,
        "Z3" : OutputLayer3
    }
    
    return Activation3, cache

def Cost(y_hat, y):
    m = y.shape[1]
    eps = 1e-8
    return -(1.0 / m) * numpy.sum(y * numpy.log(y_hat + eps))

def BackModelToFile(filename):
    numpy.savez(filename,
                W1=Weights1, W2=Weights2,
                W3=Weights3, B1=Bias1,
                B2=Bias2, B3=Bias3)
    
    print(f"Model saved to filename {filename}.npz")

def BackPropLayer3(y_hat, Activation2, Weights3, TestData):
    Activation3 = y_hat

    m = TestData.shape[1]
    
    dc_dz3 = (Activation3 - TestData) / m

    dz3_dw3 = Activation2

    dc_dw3 = dc_dz3 @ dz3_dw3.T

    dc_db3 = numpy.sum(dc_dz3, axis=1, keepdims=True)

    dc_da2 = Weights3.T @ dc_dz3

    return dc_dw3, dc_db3, dc_da2

def BackPropLayer2(propagator_dc_da2, Activation1, OutputLayer2, Weights2):
    
    da2_dz2 = reluDerivative(OutputLayer2)

    dc_dz2 = propagator_dc_da2 * da2_dz2

    dz2_dw2 = Activation1

    dc_dw2 = dc_dz2 @ dz2_dw2.T

    dc_db2 = numpy.sum(dc_dz2, axis=1, keepdims=True)
    
    dz2_da1 = Weights2
    dc_da1 = dz2_da1.T @ dc_dz2

    return dc_dw2, dc_db2, dc_da1

def BackpropLayer1(propagator_dc_da1, Activation0, OutputLayer1, Weights1):
    da1_dz1 = reluDerivative(OutputLayer1)

    dc_dz1 = propagator_dc_da1 * da1_dz1

    dz1_dw1 = Activation0

    dc_dw1 = dc_dz1 @ dz1_dw1.T

    dc_db1 = numpy.sum(dc_dz1, axis=1, keepdims=True)

    return dc_dw1, dc_db1

def train(x_train, y_train, x_val, y_val, epochs = 100, batch_size = 64, alpha = 0.01):
    
    global Weights1, Weights2, Weights3, Bias1, Bias2, Bias3
    m = x_train.shape[1]

    costs = []

    for e in range(epochs):


        perm = numpy.random.permutation(m)
        x_shuff = x_train[:, perm]
        y_shuff = y_train[:, perm]

        for i in range(0, m, batch_size):
            
            x_batch = x_shuff[:, i:i+batch_size]
            y_batch = y_shuff[:, i:i+batch_size]

            y_hat, cache = FeedForward(x_batch)

            dc_dw3, dc_db3, dc_da2 = BackPropLayer3(y_hat, cache["A2"], Weights3, y_batch)

            dc_dw2, dc_db2, dc_da1 = BackPropLayer2(dc_da2, cache["A1"], cache["Z2"], Weights2)

            dc_dw1, dc_db1 = BackpropLayer1(dc_da1, cache["A0"], cache["Z1"], Weights1)

            Weights3 = Weights3 - (alpha * dc_dw3)
            Weights2 = Weights2 - (alpha * dc_dw2)
            Weights1 = Weights1 - (alpha * dc_dw1)

            Bias3 = Bias3 - (alpha * dc_db3)
            Bias2 = Bias2 - (alpha * dc_db2)
            Bias1 = Bias1 - (alpha * dc_db1)

        train_hat, _ = FeedForward(x_train)
        val_hat, _ = FeedForward(x_val)

        train_cost = Cost(train_hat, y_train)
        val_cost = Cost(val_hat, y_val)

        train_acc = numpy.mean(numpy.argmax(train_hat, axis=0) == numpy.argmax(y_train, axis=0))
        val_acc = numpy.mean(numpy.argmax(val_hat, axis=0) == numpy.argmax(y_val, axis=0))

        print(f"epoch {e}: train_cost={train_cost:.4f}, val_cost={val_cost:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        
    return costs

x_train, y_train, x_test, y_test = prepareData()
costs = train(x_train, y_train, x_test, y_test)

BackModelToFile("model")
