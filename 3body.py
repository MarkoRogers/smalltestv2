import random

import matplotlib.pyplot as plt
import pandas as pd
import pygame
import numpy as np
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.color = color
        self.trail = []

    def compute_acceleration(self, other_bodies, G=1.0, softening=0.2):
        acceleration = np.zeros(2)
        for other in other_bodies:
            if other is not self:
                r = other.position - self.position
                distance = np.linalg.norm(r)
                acceleration += G * other.mass * r / np.maximum(distance ** 3, softening ** 3)
        return acceleration

    def draw(self, frame, system, positions, win, fade_duration, dt):
        scale = 60
        x = positions[frame, system.bodies.index(self), 0] * scale + 500
        y = 1000 - (positions[frame, system.bodies.index(self), 1] * scale + 500)
        pygame.draw.circle(win, self.color, (x, y), 5)

        # Append current position to the trail
        self.trail.append((x, y, 1.0))  # 1.0 is the initial opacity

        # Update and draw the trail
        faded_trail = []
        for pos in self.trail:
            # Reduce the opacity based on the time elapsed
            opacity = max(0, pos[2] - dt / fade_duration)
            faded_trail.append((pos[0], pos[1], opacity))
            if opacity > 0:
                faded_color = (int(self.color[0] * opacity), int(self.color[1] * opacity), int(self.color[2] * opacity))
                pygame.draw.circle(win, faded_color, (pos[0], pos[1]), 3)

        self.trail = faded_trail

    def get_state(self):
        state = np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1]])
        return state


class System:
    def __init__(self, G=1.0, state=None, bodies=None):
        self.G = G
        self.bodies = bodies

    def compute_accelerations(self):
        accelerations = []
        for body in self.bodies:
            other_bodies = [b for b in self.bodies if b is not body]
            accelerations.append(body.compute_acceleration(other_bodies, self.G))
        return accelerations

    def compute_total_energy(self):
        kinetic_energy = 0.5 * sum(body.mass * np.dot(body.velocity, body.velocity) for body in self.bodies)
        potential_energy = 0
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i + 1:]:
                distance = np.linalg.norm(body2.position - body1.position)
                potential_energy -= self.G * body1.mass * body2.mass / distance
        total_energy = kinetic_energy + potential_energy
        return total_energy

    def get_state(self):
        state = []
        for body in self.bodies:
            body_state = body.get_state()
            state.extend(body_state)
        return np.array(state)

    def integrate(self, dt, num_steps, save_positions=False):
        if save_positions:
            positions = np.zeros((num_steps, len(self.bodies), 2))
            delta_energy = np.zeros(num_steps)
            initial_energy = self.compute_total_energy()

        for step in range(num_steps):
            accelerations = self.compute_accelerations()

            for i, body in enumerate(self.bodies):
                body.position += body.velocity * dt + 0.5 * accelerations[i] * dt ** 2

            new_accelerations = self.compute_accelerations()

            for i, body in enumerate(self.bodies):
                body.velocity += 0.5 * (accelerations[i] + new_accelerations[i]) * dt

            if save_positions:
                positions[step] = [body.position for body in self.bodies]
                delta_energy[step] = initial_energy - self.compute_total_energy()

        if save_positions:
            return positions, delta_energy
        else:
            return self.get_state()


def generate():
    # Assign unique colors to each body
    body1 = Body(mass=1.0, position=[0.97000436, -0.24308753], velocity=[0.466203685, 0.43236573], color=(255, 0, 0))
    body2 = Body(mass=1.0, position=[-0.97000436, 0.24308753], velocity=[0.466203685, 0.43236573], color=(0, 255, 0))
    body3 = Body(mass=1.0, position=[0.0, 0.0], velocity=[-0.93240737, -0.86473146], color=(0, 0, 255))

    return System(bodies=[body1, body2, body3])


def pygame_run(initial):
    system = generate()
    #system = System(state=initial)

    num_steps = 6000
    dt = 0.01
    fade_duration = 1.5  # Duration for the trail to fade out in seconds
    positions, delta_energy = system.integrate(dt, num_steps, save_positions=True)

    class Canvas:
        def __init__(self):
            self.canvasX = 0
            self.canvasY = 0

        def move(self, dx, dy):
            self.canvasX += dx
            self.canvasY += dy

    canvas = Canvas()
    WIN = pygame.display.set_mode((1000, 1000))
    for i in range(positions.shape[0]):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        WIN.fill((10, 10, 10))
        for body in system.bodies:
            body.draw(i, system, positions, WIN, fade_duration, dt)

        pygame.display.flip()
        pygame.time.delay(10)


def plot_energy(delta_energy):
    plt.plot(range(len(delta_energy)), delta_energy)
    plt.xlabel('Step')
    plt.ylabel('Energy Difference')
    plt.title('Energy Conservation Over Time')
    plt.show()


def animate_matlab(positions):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('3-Body Problem Trajectories')

    line1, = ax.plot([], [], label='Body 1')
    line2, = ax.plot([], [], label='Body 2')
    line3, = ax.plot([], [], label='Body 3')
    ax.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def update(frame):
        line1.set_data(positions[:frame, 0, 0], positions[:frame, 0, 1])
        line2.set_data(positions[:frame, 1, 0], positions[:frame, 1, 1])
        line3.set_data(positions[:frame, 2, 0], positions[:frame, 2, 1])
        return line1, line2, line3

    ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=10)

    plt.show()


def make_state():
    pos1 = [0.97000436, -0.24308753]
    pos2 = [-0.97000436, 0.24308753]
    pos3 = [0.0, 0.0]
    vel1 = [0.466203685, 0.43236573]
    vel2 = [0.466203685, 0.43236573]
    vel3 = [-0.93240737, -0.86473146]

    final_state = pos1 + vel1 + pos2 + vel2 + pos3 + vel3
    return np.array(final_state)

def lyapunov(stan_state, total_time=20, delta_t=0.01, divisor=10):
    time = np.arange(0, total_time, delta_t * divisor)
    lyapunov_exponents = []

    # Create standard system
    stan_system = System(state=stan_state)

    # Create perturbed system with a small perturbation
    perturb_state = stan_state + np.random.normal(0, 1e-10, stan_state.shape)
    perturb_system = System(state=perturb_state)

    # calculate initial distance between perturbed and standard system
    distance_initial = np.linalg.norm(stan_state - perturb_state)

    for step, t in enumerate(time):
        stan_state = stan_system.integrate(delta_t, divisor)
        perturb_state = perturb_system.integrate(delta_t, divisor)
        distance_final = np.linalg.norm(stan_state - perturb_state)
        lyapunov_exponent = np.log(abs(distance_final / distance_initial))
        lyapunov_exponents.append(lyapunov_exponent)

        # Renormalize the perturbation to prevent overflow
        perturb_state = stan_state + distance_initial * (perturb_state - stan_state) / distance_final
        perturb_system = System(state=perturb_state)
        distance_initial = distance_final

    # sum up the exponents and divide by total time to get average exponent
    return np.sum(lyapunov_exponents) * (1 / total_time)


def lyapunov_heatmap():
    d_values = np.linspace(0.3, 1.0, 3)
    vx_values = np.linspace(0.3, 2.0, 3)
    vy_values = np.linspace(0.3, 2.0, 3)

    results_list = []

    total_iterations = len(d_values) * len(vx_values) * len(vy_values)

    for d in tqdm(d_values, desc="d values", total=total_iterations):
        for vx in vx_values:
            for vy in vy_values:
                initial_state = make_state(d, vx, vy)
                lyapunov_exponent = lyapunov(initial_state, total_time=20)
                results_list.append({"d": d, "vx": vx, "vy": vy, "lyapunov_exponent": lyapunov_exponent})

    results = pd.DataFrame(results_list)
    results.to_csv("lyapunov_exponents_3d.csv", index=False)

pygame_run(make_state())
