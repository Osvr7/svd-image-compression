import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load the image and convert to grayscale
img = Image.open('your_image.jpg').convert('L') 
A = np.array(img)

# 2. Perform SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# 3. Define how many singular values to keep (k)
# Try different values like 5, 20, 50, 100
k_values = [5, 20, 50, 100]

plt.figure(figsize=(15, 5))

for i, k in enumerate(k_values):
    # Reconstruct the image using only 'k' components
    # we take first k columns of U, first k values of S, first k rows of Vt
    Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    plt.subplot(1, len(k_values), i+1)
    plt.imshow(Ak, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')

plt.show()

# 4. Calculate Compression Ratio for the last k
original_size = A.shape[0] * A.shape[1]
compressed_size = k * (A.shape[0] + A.shape[1] + 1)
print(f"Original size: {original_size} numbers")
print(f"Compressed size (k={k}): {compressed_size} numbers")
print(f"Efficiency: {round((1 - compressed_size/original_size)*100, 2)}% reduction")