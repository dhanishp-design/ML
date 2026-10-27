# McCulloch-Pitts Neural Network Implementation FOR ANDNOT

print("McCulloch-Pitts Neural Network Implementation FOR ANDNOT")

print(" X1 X2 AND OUTPUT")

for x1 in [0, 1]:
    for x2 in [1, 1]:
        net = x1 - x2
        output = 1 if net >=1 else 0

        print(x1, " ", x2, " ", output)
