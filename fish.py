import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from itertools import product
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Constants for the 3D grid and chunk system
ROWS, COLUMNS, DEPTH = 300, 300, 300
CHUNK_SIZE = 50  # Size of each chunk
FISH_SIZE = 5  # Uniform size for all fish
SPEED = 4  # Uniform speed for all fish
NEIGHBOR_RADIUS = 60  # Radius to consider other fish as neighbors
SEPARATION_WEIGHT = 3
ALIGNMENT_WEIGHT = 4
COHESION_WEIGHT = 4
BOUNDARY_AVOIDANCE_WEIGHT = 7.0  # Weight for boundary avoidance
OBSTACLE_AVOIDANCE_WEIGHT = 8.0  # Weight for obstacle avoidance

# Calculate the number of chunks along each axis
NUM_CHUNKS_X = ROWS // CHUNK_SIZE
NUM_CHUNKS_Y = COLUMNS // CHUNK_SIZE
NUM_CHUNKS_Z = DEPTH // CHUNK_SIZE


# Obstacle class
class Obstacle:
    def __init__(self, position, size):
        self.position = np.array(position)
        self.size = size

    def intersects(self, point):
        """Check if the given point is inside the obstacle."""
        return np.all(np.abs(self.position - point) <= self.size / 2)


# Fish class implementing Boid behavior
class Fish:
    def __init__(self, obstacles):
        self.position = np.array([random.uniform(0, ROWS), random.uniform(0, COLUMNS), random.uniform(0, DEPTH)])
        self.velocity = np.random.rand(3) * 2 - 1  # Random initial velocity
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * SPEED  # Normalize and apply speed
        self.chunk = self.get_chunk()
        self.obstacles = obstacles

    def get_chunk(self):
        """Determine the chunk index of the fish based on its position."""
        chunk_x = int(self.position[0] // CHUNK_SIZE)
        chunk_y = int(self.position[1] // CHUNK_SIZE)
        chunk_z = int(self.position[2] // CHUNK_SIZE)
        return (chunk_x, chunk_y, chunk_z)

    def update_chunk(self):
        """Update the chunk index if the fish moves to a new chunk."""
        new_chunk = self.get_chunk()
        if new_chunk != self.chunk:
            self.chunk = new_chunk

    def apply_boid_rules(self, fish_list):
        separation = np.zeros(3)
        alignment = np.zeros(3)
        cohesion = np.zeros(3)
        boundary_avoidance = np.zeros(3)
        obstacle_avoidance = np.zeros(3)
        total_neighbors = 0

        # Iterate over the chunk and adjacent chunks
        for dx, dy, dz in product([-1, 0, 1], repeat=3):
            neighbor_chunk = (self.chunk[0] + dx, self.chunk[1] + dy, self.chunk[2] + dz)
            if all(0 <= n < num for n, num in zip(neighbor_chunk, [NUM_CHUNKS_X, NUM_CHUNKS_Y, NUM_CHUNKS_Z])):
                for other_fish in fish_list:
                    if other_fish.chunk == neighbor_chunk:
                        distance = np.linalg.norm(self.position - other_fish.position)
                        if distance < NEIGHBOR_RADIUS:
                            # Separation: steer to avoid crowding neighbors
                            if distance > 0:
                                separation += (self.position - other_fish.position) / distance

                            # Alignment: steer towards the average heading of neighbors
                            alignment += other_fish.velocity

                            # Cohesion: steer towards the average position of neighbors
                            cohesion += other_fish.position

                            total_neighbors += 1

        if total_neighbors > 0:
            # Average the alignment and cohesion vectors
            alignment /= total_neighbors
            cohesion /= total_neighbors

            # Calculate the cohesion vector
            direction_to_cohesion = cohesion - self.position
            norm_direction_to_cohesion = np.linalg.norm(direction_to_cohesion)
            if norm_direction_to_cohesion > 0:
                cohesion = direction_to_cohesion / norm_direction_to_cohesion
            else:
                cohesion = np.zeros(3)  # Avoid division by zero if there's no direction

            # Apply weights to each of the steering behaviors
            separation *= SEPARATION_WEIGHT
            alignment *= ALIGNMENT_WEIGHT
            cohesion *= COHESION_WEIGHT

            # Combine the steering behaviors
            self.velocity += separation + alignment + cohesion

        # Boundary avoidance logic
        boundary_avoidance += self.avoid_boundaries()
        self.velocity += boundary_avoidance

        # Obstacle avoidance logic
        obstacle_avoidance += self.avoid_obstacles()
        self.velocity += obstacle_avoidance

        self.velocity = self.velocity / np.linalg.norm(self.velocity) * SPEED

    def avoid_boundaries(self):
        """Steers the fish away from the edges of the tank."""
        avoidance = np.zeros(3)
        for i, pos in enumerate(self.position):
            if pos < 5:
                avoidance[i] = 1 / (pos + 0.1)  # Avoid the lower boundary
            elif pos > [ROWS, COLUMNS, DEPTH][i] - 5:
                avoidance[i] = -1 / ([ROWS, COLUMNS, DEPTH][i] - pos + 0.1)  # Avoid the upper boundary

        return avoidance * BOUNDARY_AVOIDANCE_WEIGHT

    def avoid_obstacles(self):
        """Steers the fish away from obstacles."""
        avoidance = np.zeros(3)
        for obstacle in self.obstacles:
            distance_to_obstacle = np.linalg.norm(self.position - obstacle.position)
            if distance_to_obstacle < obstacle.size:
                direction_to_obstacle = (self.position - obstacle.position) / distance_to_obstacle
                avoidance += direction_to_obstacle / distance_to_obstacle

        return avoidance * OBSTACLE_AVOIDANCE_WEIGHT

    def move(self, fish_list):
        self.apply_boid_rules(fish_list)
        new_position = self.position + self.velocity

        # Ensure the fish stays within bounds
        new_position = np.clip(new_position, [0, 0, 0], [ROWS - 1, COLUMNS - 1, DEPTH - 1])
        self.position = new_position
        self.update_chunk()


# Initialize obstacles
obstacles = [
    Obstacle(position=[100, 100, 100], size=20),
    Obstacle(position=[200, 200, 200], size=30),
    Obstacle(position=[150, 150, 150], size=25)
]

# Initialize multiple fish with uniform size and speed
fish_list = [Fish(obstacles) for _ in range(600)]

# Plotting setup
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Stylization parameters
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.grid(False)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

ARROW_THICKNESS = 1
ARROW_COLOR = 'blue'
OBSTACLE_COLOR = 'red'


def update(frame):
    ax.clear()
    ax.set_xlim(0, ROWS - 1)
    ax.set_ylim(0, COLUMNS - 1)
    ax.set_zlim(0, DEPTH - 1)
    ax.set_facecolor('black')

    for fish in fish_list:
        fish.move(fish_list)
        pos = fish.position
        vel = fish.velocity
        norm_vel = vel / np.linalg.norm(vel)
        ax.quiver(pos[0], pos[1], pos[2], norm_vel[0], norm_vel[1], norm_vel[2],
                  length=10.0, color=ARROW_COLOR, arrow_length_ratio=0.5, linewidth=ARROW_THICKNESS)

    # Plot obstacles
    for obstacle in obstacles:
        r = obstacle.size / 2
        x, y, z = obstacle.position
        u, v, w = np.meshgrid(
            np.linspace(-r, r, 10),
            np.linspace(-r, r, 10),
            np.linspace(-r, r, 10)
        )
        u = u.flatten()
        v = v.flatten()
        w = w.flatten()
        points = np.vstack((u, v, w)).T + obstacle.position
        in_box = np.all(np.abs(points - obstacle.position) <= r, axis=1)
        ax.scatter(points[in_box][:, 0], points[in_box][:, 1], points[in_box][:, 2],
                   color=OBSTACLE_COLOR, s=10)

    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title(f'Fish Count: {len(fish_list)}', color='white')


# Parameters for saving the video
total_frames = 300  # Number of frames
frame_rate = 60  # Frames per second

# Create the animation
ani = FuncAnimation(fig, update, frames=total_frames, blit=False)

# Save the animation as a video file
writer = FFMpegWriter(fps=frame_rate, metadata=dict(artist='Me'), bitrate=3200)
ani.save("boid_simulation_with_obstacles.mp4", writer=writer)

# Close the plot
plt.close()
