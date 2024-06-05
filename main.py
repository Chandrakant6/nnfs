from nnfs import *

X, y = vertical_data(100,3)

l1 =Layer(2, 3)
act1 = ReLU()

l2 = Layer(3, 3)
act2 = ReLU()

l1.forward(X)
act1.forward(l1.output)

l2.forward(act1.output)
act2.forward(l2.output)

# print(atc2.output[:5])

loss_func = CCE()
loss = loss_func.calculate(act2.output, y)

print(loss)