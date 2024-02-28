import numpy as np

X = np.array(([2,9],[1,5],[3,6]), dtype=float)
y = np.array(([92],[86],[89]), dtype=float)
X = X/ np.amax(X, axis=0)
y = y/100

def sigmoid(x):
    return x/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch = 1000
learning_rate = 0.6
input_layer = 2
hidden_layer = 3
output_layer = 1
wh = np.random.uniform(size=(input_layer,hidden_layer))
bh = np.random.uniform(size=(1, hidden_layer))
wo = np.random.uniform(size=(hidden_layer, output_layer))
bo = np.random.uniform(size=(1, output_layer))

for i in range(epoch):
    net_h = np.dot(X, wh) + bh
    sigma_h = sigmoid(net_h)
    net_o = np.dot(sigma_h, wo) + bo
    output = sigmoid(net_o)
    deltaK = (y-output)*derivatives_sigmoid(output)
    deltaH = deltaK.dot(wo.T)* derivatives_sigmoid(sigma_h)
    wo = wo+ sigma_h.T.dot(deltaK)*learning_rate
    wh= wh + X.T.dot(deltaH)*learning_rate

print(f"Input:\n {X}*")
print(f"Actual output:\n {y}*")
print(f"Expected output:\n {output}")