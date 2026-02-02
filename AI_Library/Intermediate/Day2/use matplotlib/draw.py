import matplotlib.pyplot as plt
import numpy as np
X=np.array([2,4,6])
Y=np.array([4,8,12])
mean_x = np.mean(X)
mean_y = np.mean(Y)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x)**2)
a = numerator / denominator
b = mean_y - (a * mean_x)
#       قيم خط الانحدار     #
Y_line = a * X + b  
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='blue', marker='o', label='data')
#plt.show()
plt.plot(X, Y_line, color='red', linestyle='-', linewidth=2, label=f' regression line (Y = {a:.2f}X + {b:.2f})')
#plt.show()

X_new = 2.5
Y_pred = a * X_new + b

plt.scatter(X_new, Y_pred, color='green', marker='X', s=200, label=f'predict {X_new} ton: {Y_pred:.2f} litter')

#plt.show()




plt.axvline(x=X_new, color='gray', linestyle='--', linewidth=0.8) 
# خط رأسي عند قيمة التنبؤ
plt.axhline(y=Y_pred, color='gray', linestyle='--', linewidth=0.8) 
# 
plt.show()
