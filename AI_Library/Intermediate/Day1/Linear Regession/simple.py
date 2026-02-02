import numpy as np
#Create x and y data

X=np.array([2,4,6])
Y=np.array([4,8,12])
mean_x = np.mean(X)

mean_y = np.mean(Y)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x)**2)
a = numerator / denominator
b = mean_y - (a * mean_x)

#print calculated values
print("a =",a)
print("b =",b)
def predict(x, a, b):
    return a * x + b

prediction = predict(3, a, b)
print(prediction)

