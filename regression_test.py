import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from random import choice, randint
import string

# Generate some synthetic data
def generate_random_word():
    return ''.join([choice(string.ascii_lowercase) for _ in range(randint(1, 2))])

hsh = lambda x: int("".join([str(ord(i.upper())) for i in list(x)]))

inwords = np.array(sorted([hsh(generate_random_word()) for _ in range(10)]))

xs = np.array(range(len(inwords)))


digits = 2
for i, v in enumerate(inwords):
    if len(str(v)) != digits:
        break

x1 = inwords[:i]
y1 = inwords[:i]
x2 = inwords[i:]
y2 = inwords[i:]

p1 = poly1d(polyfit(x1, y1, 10))
p2 = poly1d(polyfit(x2, y2, 10))


# Sigmoid function for blending
def sigmoid(x, x0=0, gamma=10):
    return 1 / (1 + np.exp(-gamma * (x - x0)))

# Blended polynomial function
def blended_polynomial(x, p1, p2, x0=0):
    blend_factor = sigmoid(x, x0)
    return (1 - blend_factor) * p1(x) + blend_factor * p2(x)

# Combine polynomials across the whole range
y_blended = blended_polynomial(inwords, p1, p2, x0=0)

# Plot the results
plt.plot(inwords, xs, label="Original Data", linestyle='dotted')
plt.plot(inwords, p1(inwords), label="Polynomial 1 (Region 1)", linestyle='--')
plt.plot(inwords, p2(inwords), label="Polynomial 2 (Region 2)", linestyle='--')
plt.plot(inwords, y_blended, label="Blended Polynomial", color='red')
plt.legend()
plt.savefig("plt.png")
