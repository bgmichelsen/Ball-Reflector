import numpy as np


class Ball:
    def __init__(self, orig=(0, 0, 0), velocity=(1, 1, 1)) -> None:
        self.orig = np.array([float(orig[0]), float(orig[1]), float(orig[2])])
        self.v = np.array([float(velocity[0]), float(velocity[1]), float(velocity[2])])
        self.g = 9.8
        self.pos = [0, 0, 0]
        self.v_prime = np.array([0.0, 0.0, 0.0])

    def update(self, timestep) -> None:
        # Get the ball position for the timestep
        self.pos[0] = self.orig[0] + self.v[0] * timestep
        self.pos[1] = self.orig[1] + self.v[1] * timestep
        self.pos[2] = self.orig[2] + self.v[2] * timestep
        self.pos[2] = self.pos[2] - (0.5 * self.g) * timestep**2

        # Get the velocity for the timestep
        self.v_prime[0] = self.v[0]
        self.v_prime[1] = self.v[1]
        self.v_prime[2] = self.v[2] - self.g * timestep

    def position(self) -> tuple:
        return (self.pos[0], self.pos[1], self.pos[2])

    def velocity(self) -> tuple:
        return (self.v_prime[0], self.v_prime[1], self.v_prime[2])