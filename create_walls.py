import imageio
import numpy as np


SIZE_X = 160
SIZE_Y = 400

image = np.zeros([SIZE_X, SIZE_Y, 3], dtype = np.uint8) + 255

# image[5:43,5,:] = 0
# image[5:43,59,:] = 0
# image[5,5:59,:] = 0
# image[42,5:59,:] = 0

imageio.imwrite("images/walls.png", image)
