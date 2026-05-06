import numpy as np
import matplotlib.pyplot as plt

class Checker:

    def __init__(self, resolution, tile_size):

        if tile_size <= 0:
            raise ValueError("size must be positive")
        
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        
        if resolution % 2 != 0:
            raise ValueError("resolution must be even")

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        
        zeros = np.zeros((self.tile_size, self.tile_size))
        ones  = np.ones((self.tile_size, self.tile_size))

        top = np.hstack((zeros, ones)) # horizontal stack of zeros and ones to create the top half of the tile
        bottom = np.hstack((ones, zeros)) 

        tile = np.vstack((top, bottom)) # vertical stack of top and bottom to create the full tile

        repetitions = self.resolution // (2 * self.tile_size)

        self.output = np.tile(tile, (repetitions, repetitions))

        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.show()
        

class Circle:
    
    def __init__(self, resolution, radius, position):

        if resolution <= 0:
            raise ValueError("resolution must be positive")

        if radius <= 0:
            raise ValueError("radius must be positive")

        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)

        xx, yy = np.meshgrid(x, y) #creates two matrices of shape (resolution, resolution) where xx contains the x coordinates and yy contains the y coordinates of each pixel

        center_x, center_y = self.position

        distance = (xx - center_x) ** 2 + (yy - center_y) ** 2 #matrix of distances from the center

        self.output = distance <= self.radius ** 2 #boolean matrix where True is inside the circle and False is outside
        return self.output.copy()

    
    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.show()

class Spectrum:
    
    def __init__(self, resolution):

        if resolution <= 0:
            raise ValueError("resolution must be positive")

        self.resolution = resolution
        self.output = None

    def draw(self):
        x = np.linspace(0, 1, self.resolution) #creates a vector of shape (resolution,) with values from 0 to 1
        y = np.linspace(0, 1, self.resolution)

        xx, yy = np.meshgrid(x, y) # creates two matrices of shape (resolution, resolution) where xx contains the red values and yy contains the green values of each pixel

        R = xx #red channel is the x coordinate
        G = yy #green channel is the y coordinate
        B = 1 - xx #blue channel is the complement of the x coordinate

        self.output = np.stack([R, G, B], axis=-1) #stack the channels to create an RGB image
        return self.output.copy()

    
    def show(self):
        plt.imshow(self.output)
        plt.axis("off")
        plt.show()

    

