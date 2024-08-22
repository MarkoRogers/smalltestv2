import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors


# simulation class for wildfires
class WFSim:
    def __init__(self, f=0.01, p=1e-4, wind='NE', bedrock=0.005, water=0.05, grass=0.1, cloud=0.1, h=16, w=16):
        # initializing parameters
        self.f = f  # probability of fire spreading
        self.p = p  # probability of tree growth
        self.h = h  # height of the landscape
        self.w = w  # width of the landscape
        self.bedrock = bedrock  # probability of bedrock presence
        self.water = water  # probability of water presence
        self.wind = wind  # wind direction affecting fire spread
        self.cloud = np.random.random(1)[0]  # initial cloud coverage
        self.temp = self.temperature()  # temperature throughout the day

        # defining fire spread directions based on wind
        self.offsets = {'calm': [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
                        'N': [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)],
                        'S': [(0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
                        'E': [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1)],
                        'W': [(-1, -1), (-1, 0), (0, -1), (1, -1), (1, 0)],
                        'NE': [(-1, 0), (-1, 1), (0, 1)],
                        'NW': [(-1, -1), (-1, 0), (0, -1)],
                        'SE': [(0, 1), (1, 0), (1, 1)],
                        'SW': [(0, -1), (1, -1), (1, 0)]}

        # defining cloud movement based on wind
        self.cloud_offsets = {'calm': [],
                              'N': [(1, 0)],
                              'S': [(-1, 0)],
                              'E': [(0, -1)],
                              'W': [(0, 1)],
                              'NE': [(1, -1)],
                              'NW': [(1, 1)],
                              'SE': [(-1, -1)],
                              'SW': [(-1, 1)]}

        # initializing landscape with random trees and empty spaces
        self.landscape = np.random.randint(0, 2, (self.h, self.w))  # 0 = empty, 1 = tree
        self.old_landscape = self.landscape.copy()  # copy of previous state

        # calculating initial burned ratio and tree cover
        self.burned_ratio = round((self.landscape == -1).sum() / (self.h * self.w) * 100, 2)
        self.tree_cover = round((self.landscape == 1).sum() / (self.h * self.w) * 100, 2)

        # adding bedrock and water features to the landscape
        for i in range(self.landscape.shape[0]):
            for j in range(self.landscape.shape[1]):
                coef = 4 if self.surf_neighbors_check(i, j, "B") else 1
                if self.bedrock * coef > np.random.random():
                    self.landscape[i, j] = -2  # bedrock

                coef = 10 if self.surf_neighbors_check(i, j, "W") else 0.1
                if self.water * coef > np.random.random():
                    self.landscape[i, j] = -3  # water

    # checking if any neighbor cell is on fire
    def fire_neighbors_check(self, idx, jdx):
        check = False  # initialize fire check to false
        offsets = self.offsets[self.wind]  # get wind direction offsets

        for di, dj in offsets:  # check neighboring cells
            ni, nj = idx + di, jdx + dj
            if nj >= 0 and ni >= 0 and ni < self.h and nj < self.w:
                if self.landscape[ni, nj] == 2 and self.landscape[idx, jdx] == 1:
                    check += True  # fire spreads if conditions are met
        return check

    # checking if any neighboring cell has specific terrain feature (water or bedrock)
    def surf_neighbors_check(self, idx, jdx, kind='W'):
        if kind == 'W':
            value = -2  # bedrock value
        elif kind == 'B':
            value = -1  # burned tree value
        check = False
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for di, dj in offsets:
            ni, nj = idx + di, jdx + dj
            if nj >= 0 and ni >= 0 and ni < self.h and nj < self.w:
                if self.landscape[ni, nj] == value:
                    check += True
        return check

    # simulate temperature over a 24-hour period
    def temperature(self, average_temp=20, amplitude=5, noise_level=2):
        hours = np.arange(24)  # 24-hour range
        temperatures = average_temp + amplitude * np.sin(2 * np.pi * hours / 24 - np.pi / 2)  # temp cycle

        temperatures += np.random.normal(0, noise_level, 24)  # add noise for realism

        return temperatures

    # generate cloud cover
    def generate_cloud(self):
        size = 16  # arbitrary cloud size
        mask = np.zeros((self.h, self.w))  # initialize mask for cloud
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        idx_prev, jdx_prev = np.random.randint(0, self.h), np.random.randint(0, self.w)
        for _ in range(size):
            for di, dj in [offsets[np.random.randint(len(offsets))]]:
                ni, nj = idx_prev + di, jdx_prev + dj
                if nj >= 0 and ni >= 0 and ni < self.h and nj < self.w:
                    cell = (np.random.randint(min(0, idx_prev - 1), min(self.h, idx_prev + 1)),
                            np.random.randint(min(0, jdx_prev - 1), min(self.w, jdx_prev + 1)))
                    mask[ni, nj] = 1  # mark cloud area
                idx_prev, jdx_prev = ni, nj
        return mask.astype(bool)

    # simulate cloud movement based on wind direction
    def cloud_move(self):
        offsets = self.cloud_offsets[self.wind]
        mask = np.zeros((self.h, self.w))
        for idx in range(self.landscape.shape[0]):
            for jdx in range(self.landscape.shape[1]):
                for di, dj in offsets:
                    ni, nj = idx + di, jdx + dj
                    if nj >= 0 and ni >= 0 and ni < self.h and nj < self.w and self.landscape[idx, jdx] == 3:
                        mask[ni, nj] = 1  # move cloud based on wind
        return mask.astype(bool)

    # step function to update landscape state
    def step(self, step):

        if step % 24 == 0 and step > 0:
            self.temp = self.temperature()  # update temperature every 24 steps

        new_landscape = self.landscape.copy()  # create a copy of the current landscape

        for i in range(self.landscape.shape[0]):
            for j in range(self.landscape.shape[1]):

                if new_landscape[i, j] == 3:
                    if self.old_landscape[i, j] == -1 or self.old_landscape[i, j] == 2:
                        new_landscape[i, j] = 0  # cloud over burned or burning area becomes empty
                    else:
                        new_landscape[i, j] = self.old_landscape[i, j]

                if new_landscape[i, j] == 2:
                    new_landscape[i, j] = -1  # burning trees become burned trees
                if self.p > np.random.rand() and self.landscape[i, j] == 0:
                    new_landscape[i, j] = 1  # empty land grows new trees

                coef = 2 if self.temp[step % 24] > 25 else 1  # increase fire chance if temperature is high
                if (self.f * coef > np.random.rand() or self.fire_neighbors_check(i, j)) and self.landscape[i, j] == 1:
                    new_landscape[i, j] = 2  # tree catches fire

        self.old_landscape = new_landscape.copy()  # save current state as old state for next step

        if 3 in self.landscape and self.wind != 'calm':
            new_landscape[self.cloud_move()] = 3  # move cloud if wind isn't calm

        if (self.cloud > np.random.rand()):
            new_landscape[self.generate_cloud()] = 3  # generate new cloud randomly

        self.landscape = new_landscape.copy()  # update landscape with new state
        self.burned_ratio = round((self.landscape == -1).sum() / (self.h * self.w) * 100, 2)  # update burned ratio
        self.tree_cover = round((self.landscape == 1).sum() / (self.h * self.w) * 100,
                                2)  # update tree cover percentage

    # Method to create a custom map from an mxn array
    def set_custom_map(self, custom_map):
        assert custom_map.shape == (self.h, self.w), "Custom map must match the initialized landscape dimensions."
        self.landscape = custom_map.copy()  # update the landscape with the custom map

    # Method to create a custom map from a picture (sampling colors)
    def set_map_from_image(self, image_array):
        assert image_array.shape[:2] == (self.h, self.w), "Image must match the initialized landscape dimensions."
        # Assuming the image has RGB channels, convert to grayscale and threshold to define terrain
        gray_image = np.mean(image_array, axis=2)  # average the RGB channels to get a grayscale image
        self.landscape = np.zeros_like(gray_image)

        self.landscape[gray_image < 50] = -3  # water
        self.landscape[(gray_image >= 50) & (gray_image < 100)] = -2  # bedrock
        self.landscape[(gray_image >= 100) & (gray_image < 150)] = 0  # empty land
        self.landscape[(gray_image >= 150) & (gray_image < 200)] = 1  # trees
        self.landscape[gray_image >= 200] = 3  # clouds


# Function to update the animation frame by frame
def update(frame):
    im.set_data(Sim.landscape)  # update the image with the new landscape state
    ax.axis('off')  # hide the axis
    Sim.step(frame)  # simulate the next step
    return [im]


# Define colors for each type of terrain or condition
colors_list = ['steelblue', 'grey', 'black', 'olivedrab', 'forestgreen', 'orange', 'white']  # color map
cmap = colors.ListedColormap(colors_list)  # create the color map
bounds = [-3, -2, -1, 0, 1, 2, 3, 4]  # define boundaries for the colors
norm = colors.BoundaryNorm(bounds, cmap.N)  # normalize the colors

Sim = WFSim(h=256, w=256)  # initialize the simulation with a 24x24 grid

fig, ax = plt.subplots(figsize=(16, 16))  # create a figure
im = ax.imshow(Sim.landscape, cmap=cmap, norm=norm)  # display the initial landscape
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # adjust layout

ani = animation.FuncAnimation(fig, update, frames=720, interval=20)  # create the animation, simulating 100 steps
ani.save('fire.gif', fps=24, savefig_kwargs={'pad_inches': 0})  # save the animation as a gif
