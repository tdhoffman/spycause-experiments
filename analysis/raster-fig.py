import numpy as np
import matplotlib.pyplot as plt

img = np.array([[1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1]])
img = 1 - img  # flip so the colors look nice (green = treated)

plt.imshow(img, cmap="Accent")
plt.axis('off')
plt.vlines([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ymin=-0.5, ymax=4.5, colors='k')
plt.hlines([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], xmin=-0.5, xmax=4.5, colors='k')
plt.show()

pad = 0.05
plt.imshow(img, cmap="Accent")
plt.axis('off')
plt.vlines([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ymin=-0.5, ymax=4.5, colors='k')
plt.hlines([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], xmin=-0.5, xmax=4.5, colors='k')
plt.arrow(1-pad, 2, 1, 0, length_includes_head=True, head_width=0.1, color="ivory")
plt.arrow(3+pad, 2, -1, 0, length_includes_head=True, head_width=0.1, color="ivory")
plt.arrow(2, 1-pad, 0, 1, length_includes_head=True, head_width=0.1, color="ivory")
plt.arrow(2, 3+pad, 0, -1, length_includes_head=True, head_width=0.1, color="ivory")
plt.show()
