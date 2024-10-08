from numba import cuda
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the left and right images and convert to grayscale
left_image = Image.open('man-left.png').convert('L')
right_image = Image.open('man-right.png').convert('L')

left_array = np.array(left_image)
right_array = np.array(right_image)

# Output disparity map
disparity_map = np.zeros_like(left_array)

block_size = 5
max_disparity = 64 


@cuda.jit
def compute_disparity(left_img, right_img, disparity_out, block_size, max_disparity):
    x, y = cuda.grid(2)
    
    height, width = left_img.shape
    
    if x >= height or y >= width:
        return

    best_offset = 0
    min_ssd = float('inf')  # Sum of squared differences (SSD)

    # Block matching for each pixel in the left image
    for disparity in range(max_disparity):
        ssd = 0
        
        # Define block matching limits (avoid going outside the image boundaries)
        for i in range(-block_size // 2, block_size // 2 + 1):
            for j in range(-block_size // 2, block_size // 2 + 1):
                left_x = x + i
                left_y = y + j
                
                right_x = x + i
                right_y = (y - disparity) + j
                
                if (0 <= left_x < height and 0 <= left_y < width) and (0 <= right_x < height and 0 <= right_y < width):
                    diff = int(left_img[left_x, left_y]) - int(right_img[right_x, right_y])
                    ssd += diff * diff

        # If SSD is better (smaller), update best_offset
        if ssd < min_ssd:
            min_ssd = ssd
            best_offset = disparity

    disparity_out[x, y] = best_offset  # Store disparity for this pixel

threads_per_block = (16, 16)
blocks_per_grid_x = int(np.ceil(left_array.shape[0] / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(left_array.shape[1] / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

d_left_img = cuda.to_device(left_array)
d_right_img = cuda.to_device(right_array)
d_disparity_map = cuda.to_device(disparity_map)

# Launch the kernel
compute_disparity[blocks_per_grid, threads_per_block](d_left_img, d_right_img, d_disparity_map, block_size, max_disparity)

disparity_map = d_disparity_map.copy_to_host()

# Display the disparity map
plt.imshow(disparity_map, cmap='gray')
plt.title('Disparity Map')
plt.show()