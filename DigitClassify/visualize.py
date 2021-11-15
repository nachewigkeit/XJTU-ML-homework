import matplotlib.pyplot as plt

train = [0.2481, 0.1011, 0.0309, 0.0088, 0.0017]
test = [0.2626, 0.1254, 0.0703, 0.0544, 0.0538]
plt.xticks(range(len(train)), [10, 30, 100, 300, 1000])
plt.xlabel("Number of hidden neurons", fontsize=12)

'''
train = [0.1055, 0.088, 0.0917, 0.0907, 0.1006]
test = [0.125, 0.1107, 0.1178, 0.1212, 0.1346]
plt.xticks(range(len(train)), [1, 2, 3, 4, 5])
plt.xlabel("Number of hidden layers", fontsize=12)
'''

plt.ylabel("Loss", fontsize=12)
plt.plot(range(len(train)), train, label="train")
plt.plot(range(len(test)), test, label="test")
plt.legend()
plt.show()
