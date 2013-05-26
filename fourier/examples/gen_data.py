import numpy as np
import matplotlib.pyplot as plt

# SIN
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(x)
np.savetxt('sin.txt', np.vstack((x, y)).T)

# COS
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(x)
np.savetxt('cos.txt', np.vstack((x, y)).T)

# SINCOS
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(2 * x) + 2 * np.cos(3 * x)
np.savetxt('sincos.txt', np.vstack((x, y)).T)

# square wave
x = np.linspace(0, 1, 250)
y = np.ones_like(x)
x = np.hstack((x - 2, x - 1, x, x + 1))
y = np.hstack((y, -y, y, -y))
x = np.hstack(([-2], x, [2]))
y = np.hstack(([0], y, [0]))
np.savetxt('square_wave.txt', np.vstack((x, y)).T)

# square wave moved
np.savetxt('square_wave_moved.txt', np.vstack((x + 0.5, y + 0.5)).T)

# sawtooth wave
t = np.linspace(0, 1, 250)
x = np.hstack((t - 2, t - 1, t, t + 1))
y = np.hstack((t, t - 1, t, t - 1))
x = np.hstack(([-2], x, [2]))
y = np.hstack(([0], y, [0]))
np.savetxt('sawtooth_wave.txt', np.vstack((x, y)).T)


plt.plot(x, y)
plt.show()
