import matplotlib.pyplot as plt
from util import load_csv, plot_points

# 1. Load datasets
xa, ya = load_csv('ds1_a.csv', add_intercept=False)
xb, yb = load_csv('ds1_b.csv', add_intercept=False)

# 2. Plot Dataset A
plt.figure()
plot_points(xa, ya)
plt.title('Dataset A')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('dataset_a.png')
plt.close()

# 3. Plot Dataset B
plt.figure()
plot_points(xb, yb)
plt.title('Dataset B')
plt.xlabel('x2')
plt.ylabel('x2')
plt.savefig('dataset_b.png')
plt.close()

print("Saved dataset_a.png and dataset_b.png successfully!")