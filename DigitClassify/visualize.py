import matplotlib.pyplot as plt
import numpy as np


def lossPlot(train, test, xticks, xlabel):
    plt.xticks(range(len(train)), xticks)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.plot(range(len(train)), train, label="train")
    plt.plot(range(len(test)), test, label="test")
    plt.legend()


'''
plt.figure(figsize=(10, 5))
train = [0.2481, 0.1011, 0.0309, 0.0088, 0.0017]
test = [0.2626, 0.1254, 0.0703, 0.0544, 0.0538]
xticks = [10, 30, 100, 300, 1000]
xlabel = "Number of hidden neurons"
plt.subplot(121)
plt.title("Reasonable", fontsize=12)
lossPlot(train, test, xticks, xlabel)
train = [1.1599, 0.5312, 0.3411, 0.2891, 0.2371]
test = [1.1548, 0.5352, 0.3424, 0.3007, 0.2493]
xticks = [2, 4, 6, 8, 10]
xlabel = "Number of hidden neurons"
plt.subplot(122)
plt.title("Unreasonable", fontsize=12)
lossPlot(train, test, xticks, xlabel)
plt.show()
'''

'''
train = [0.1055, 0.088, 0.0917, 0.0907, 0.1006]
test = [0.125, 0.1107, 0.1178, 0.1212, 0.1346]
xticks = [1, 2, 3, 4, 5]
xlabel = "Number of hidden layers"
lossPlot(train, test, xticks, xlabel)
plt.show()
'''

'''
train = [0.0026, 0.0039, 0.0056, 0.0080, 0.0121]
test = [0.0535, 0.05, 0.0492, 0.0499, 0.0515]
xticks = [0.1, 0.2, 0.3, 0.4, 0.5]
xlabel = "Dropout Rate"
lossPlot(train, test, xticks, xlabel)
plt.show()
'''

'''
train = [0.008, 0.014, 0.0176, 0.021]
test = [0.0475, 0.0471, 0.0498, 0.0497]
xticks = [5, 10, 15, 20]
xlabel = "Max Rotation Angle"
lossPlot(train, test, xticks, xlabel)
plt.show()
'''

train = [0.0154, 0.0058, 0.0317, 0.014]
test = [0.0527, 0.0488, 0.0551, 0.0471]
xticks = ["no", "norm", "rotate", "norm&rotate"]

total_width, n = 0.8, 2
x = np.arange(len(train))
width = total_width / n
x = x - (total_width - width) / 2
plt.xticks(range(len(train)), xticks)
plt.xlabel("Preprocess", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.bar(x, train, width=width, label='train')
plt.bar(x + width, test, width=width, label='test')
plt.legend()
plt.show()
