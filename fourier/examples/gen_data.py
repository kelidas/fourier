import numpy as np
import matplotlib.pyplot as plt

# SIN
x = np.linspace(-np.pi, np.pi, 1000)
y = np.sin(x)
np.savetxt('sin.txt', np.vstack((x, y)).T)

# COS
x = np.linspace(-np.pi, np.pi, 1000)
y = np.sin(x)
np.savetxt('cos.txt', np.vstack((x, y)).T)

# SINCOS
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(2 * x) + 2 * np.cos(3 * x)
np.savetxt('sincos.txt', np.vstack((x, y)).T)

# square wave
x = np.linspace(-2, 2, 1000)
y = np.array([1, -1, 1, -1]).repeat(1000 / 4.)
y[0] = 0
y[-1] = 0
np.savetxt('square_wave.txt', np.vstack((x, y)).T)

# square wave moved
x = np.linspace(-2, 2, 1000)
y = np.array([1, -1, 1, -1]).repeat(1000 / 4.)
y[0] = 0
y[-1] = 0
np.savetxt('square_wave_moved.txt', np.vstack((x + 0.5, y)).T)

# sawtooth wave
x = np.linspace(-2, 2, 1000)
y = np.linspace(0, 1, 1000 / 4.)
y = np.hstack((y, y - 1, y, y - 1))
np.savetxt('sawtooth_wave.txt', np.vstack((x, y)).T)


plt.plot(x, y)
plt.show()
