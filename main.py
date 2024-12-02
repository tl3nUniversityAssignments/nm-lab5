import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Оригінальна функція
def f(x):
    return np.e ** x - 2 * (x - 1) ** 2

# Отримуємо вузли, як у попередній лабораторній
def get_nodes(a, b, m):
    nodes = np.linspace(a, b, m)
    return nodes

# Обраховуємо коефіцієнти для побудови сплайну
def compute_coefficients(x_nodes, y_nodes):
    n = len(x_nodes) - 1  # Кількість інтервалів
    h = x_nodes[1] - x_nodes[0]  # Крок

    # Створюємо тридіагональну матрицю та праву частину
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)

    # Так як ми маємо рівномірний розподіл вузлів, то можемо трішки відозмінити заповнення матриці 
    A[0, 0] = A[n, n] = 1
    for i in range(1, n):
        A[i, i-1] = h
        A[i, i] = 4 * h
        A[i, i+1] = h
        b[i] = 6 * ((y_nodes[i+1] - y_nodes[i]) / h - (y_nodes[i] - y_nodes[i-1]) / h)

    print("Матриця A:")
    sp.pprint(sp.Matrix(A))

    # Розв'язуємо систему рівнянь і знаходимо коефіцієнти c
    c = np.linalg.solve(A, b)
    print(f"Коефіцієнти c: {c}")

    a = np.zeros(n + 1)
    d = np.zeros(n + 1)
    b = np.zeros(n + 1)

    # Знаходимо коефіцієнти a, b, d
    for i in range(1, n + 1):
        a[i] = y_nodes[i]
        d[i] = (c[i] - c[i-1]) / h
        b[i] = (h / 2) * c[i] - (h ** 2 / 6) * d[i] + (y_nodes[i] - y_nodes[i-1]) / h
    
    print(f"Коефіцієнти a: {a}")
    print(f"Коефіцієнти b: {b}")
    print(f"Коефіцієнти d: {d}")

    return a, b, c, d

def natural_cubic_spline_interpolation(x, x_nodes, a, b, c, d):
    for i in range(1, len(x_nodes)):
        if x_nodes[i-1] <= x <= x_nodes[i]:
            return a[i] + b[i] * (x - x_nodes[i]) + c[i] * (x - x_nodes[i]) ** 2 + d[i] * (x - x_nodes[i]) ** 3
    return None

# Знаходимо вузли та значення функції в них
a = -1 # Ліва границя
b = 1 # Права границя
m = 10 # Кількість вузлів
nodes = get_nodes(a, b, m)
nodes_y = f(nodes)

# Обраховуємо коефіцієнти для побудови сплайну
a, b, c, d = compute_coefficients(nodes, nodes_y)

# Графік
x_vals = np.linspace(-20, 20, 1000) # Вказуємо x

# Обмежуємо графік
plt.xlim(-20, 20)
plt.ylim(-100, 100)

# Будуємо координатні осі
plt.axhline(0, color="black", lw=0.5)
plt.axvline(0, color="black", lw=0.5)

# Будуємо графіки
plt.plot(x_vals, [f(x) for x in x_vals], label="f(x)", color="blue") # Оригінальна функція
plt.plot(x_vals, [natural_cubic_spline_interpolation(x, nodes, a, b, c, d) for x in x_vals], "--", label="Сплайн", color="red") # Сплайн

# Вказуємо вузли
plt.scatter(nodes, nodes_y, label="Вузли", color="black")
plt.title("Графік функції та сплайну")
plt.grid()
plt.legend()
plt.show()