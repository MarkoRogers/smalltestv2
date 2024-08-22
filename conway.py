import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class GameOfLife:
    def __init__(self, grid):
        self.N = grid.shape[0]  # Height
        self.M = grid.shape[1]  # Width
        self.grid = grid

    def update(self):
        newGrid = self.grid.copy()
        for i in range(self.N):
            for j in range(self.M):
                # Calculate the sum of the eight neighbors
                total = int((self.grid[i, (j - 1) % self.M] + self.grid[i, (j + 1) % self.M] +
                             self.grid[(i - 1) % self.N, j] + self.grid[(i + 1) % self.N, j] +
                             self.grid[(i - 1) % self.N, (j - 1) % self.M] + self.grid[
                                 (i - 1) % self.N, (j + 1) % self.M] +
                             self.grid[(i + 1) % self.N, (j - 1) % self.M] + self.grid[
                                 (i + 1) % self.N, (j + 1) % self.M]) / 255)

                # Apply Conway's rules
                if self.grid[i, j] == 255:  # Alive
                    if (total < 2) or (total > 3):
                        newGrid[i, j] = 0  # Dies
                else:  # Dead
                    if total == 3:
                        newGrid[i, j] = 255  # Becomes alive

        self.grid[:] = newGrid[:]
        return self.grid

    def run_simulation(self, steps, save_path='conway.gif'):
        frames = []

        for _ in range(steps):
            updated_grid = self.update()
            frame_image = Image.fromarray(np.uint8(updated_grid))
            frames.append(frame_image)

        frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=1, duration=200)
        print(f"Simulation saved as {save_path}")


def load_grid_from_png(file_path):
    # Open the image file
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    # Convert image to numpy array
    grid = np.array(img)
    # Normalize to binary values: 0 for dead (white), 255 for alive (black)
    grid = np.where(grid < 128, 255, 0)  # Black is 255 (alive), white is 0 (dead)
    return grid


def main():
    # Load grid from 'conway.png'
    grid = load_grid_from_png('conway.png')
    print(f"Grid size: {len(grid)}x{len(grid[0])}")
    steps = int(input("Enter the number of simulation steps: "))

    game = GameOfLife(grid)

    # Run the simulation
    game.run_simulation(steps)


if __name__ == "__main__":
    main()
