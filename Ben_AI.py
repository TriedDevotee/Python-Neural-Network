from PIL import Image
import numpy

def load_model(filename):
    global Weights1, Weights2, Weights3, Bias1, Bias2, Bias3

    data = numpy.load(filename)
    Weights1 = data["W1"]
    Weights2 = data["W2"]
    Weights3 = data["W3"]

    Bias1 = data["B1"]
    Bias2 = data["B2"]
    Bias3 = data["B3"]

    print(f"Model Loaded from {filename}")

def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))

    arr = numpy.array(img).astype(numpy.float32)
    arr = arr.reshape(784, 1)
    return arr

def relu(z):
    return numpy.maximum(0, z)

def softmax(z):
    z_shift = z - numpy.max(z, axis=0, keepdims=True)
    expZ = numpy.exp(z_shift)
    return expZ / numpy.sum(expZ, axis=0, keepdims=True)

def FeedForward(Activation0):

    OutputLayer1 = Weights1 @ Activation0 + Bias1
    Activation1 = relu(OutputLayer1)

    OutputLayer2 = Weights2 @ Activation1 + Bias2
    Activation2 = relu(OutputLayer2)

    OutputLayer3 = Weights3 @ Activation2 + Bias3
    Activation3 = softmax(OutputLayer3)
    
    return Activation3

def predict(x):
    y_hat = FeedForward(x)
    return numpy.argmax(y_hat, axis=0)[0]

load_model("model.npz")

sample = preprocess_image("./TestImages/TestData4Number2.png")
prediction = predict(sample)

print(f"The bot thinks it is the number: {prediction}")