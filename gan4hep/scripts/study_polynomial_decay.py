# %% 
import numpy as np
import matplotlib.pyplot as plt

start = 0.001
end = 0.000001
x = np.arange(0, 1000, 1)
powers = [0.25, 0.5, 1., 2., 4]
max_steps = 1000
def decayer(power):
    return (start - end)* np.power((1 - x/max_steps), power) + end
ys = [decayer(p) for p in powers]

[plt.plot(x, y, label='power={:.2f}'.format(powers[idx])) for idx,y in enumerate(ys)]
# plt.yscale('log')
plt.legend()
plt.xlabel("epoch", fontsize=14)
plt.ylabel("learning rate", fontsize=14)
plt.title("Polynomial Scheduler")
plt.savefig("learning.png")
# %%
