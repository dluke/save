

# test particle shader code to get a conceptual understanding 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%

rows = 16
N = 16**2 + 32
x_list = []
y_list = []
for index in range(N):
	y = float(index)
	x = y % rows
	y = (y - x) / rows

	x -= rows/2
	y -= rows/2

	x_list.append(x)
	y_list.append(y)

# %%


sns.scatterplot(x_list, y_list)
plt.gca().set_aspect('equal')


# %%
16 * 16