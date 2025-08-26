# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date: 26.08.2025
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = sm.datasets.sunspots.load_pandas().data
years = data['YEAR'].tolist()
sunspots = data['SUNACTIVITY'].tolist()

X = [i - years[len(years)//2] for i in years]
x2 = [i**2 for i in X]
xy = [i*j for i, j in zip(X, sunspots)]
n = len(years)

b = (n*sum(xy) - sum(sunspots)*sum(X)) / (n*sum(x2) - (sum(X)**2))
a = (sum(sunspots) - b*sum(X)) / n
linear_trend = [a + b*X[i] for i in range(n)]

x3 = [i**3 for i in X]
x4 = [i**4 for i in X]
x2y = [i*j for i, j in zip(x2, sunspots)]
coeff = [[n, sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]
Y = [sum(sunspots), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly*X[i] + c_poly*(X[i]**2) for i in range(n)]

print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(years, sunspots, 'o', label="Sunspot Data", alpha=0.5)
plt.plot(years, linear_trend, 'r--', label="Linear Trend")
plt.xlabel("Year")
plt.ylabel("Sunspot Count")
plt.title("Sunspots - Real Data vs Linear Trend")
plt.legend()

plt.subplot(2,1,2)
plt.plot(years, sunspots, 'o', label="Sunspot Data", alpha=0.5)
plt.plot(years, poly_trend, 'g-', label="Polynomial Trend (Degree 2)")
plt.xlabel("Year")
plt.ylabel("Sunspot Count")
plt.title("Sunspots - Real Data vs Polynomial Trend (Degree 2)")
plt.legend()

plt.tight_layout()
plt.show()
```
### OUTPUT

A - LINEAR TREND ESTIMATION

<img width="1257" height="663" alt="image" src="https://github.com/user-attachments/assets/dc1c3839-a515-4c5c-a922-6afb01cfcd21" />

B- POLYNOMIAL TREND ESTIMATION

<img width="1271" height="656" alt="image" src="https://github.com/user-attachments/assets/7398f776-701f-470e-a5c8-fbab38bb1def" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
