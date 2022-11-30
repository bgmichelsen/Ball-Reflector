import numpy as np


class Backboard:
    def __init__(self, orig=(0, 0, 0), size=(10, 10)) -> None:
        self.orig = np.array([float(orig[0]), float(orig[1]), float(orig[2])])
        # Plane is initially parallel to the XZ-plane
        self.norm = np.array([float(0), float(1), float(0)])
        self.w = size[0]
        self.h = size[1]

    def draw(self) -> tuple:
        # Equation of a plane: a*x + b*y + c*z + d = 0
        # Where: [a, b, c] = the normal vector
        # d is calculated below (dot product with normal)
        # From the equation of a plane:
        # d = -(a*x + b*y + c*z)
        d = -self.orig.dot(self.norm)

        # Create the x, z axis of the plane
        x_ax = np.linspace((self.orig[0] - (self.w // 2)), (self.orig[0] + (self.w // 2)), self.w)
        z_ax = np.linspace((self.orig[2] - (self.h // 2)), (self.orig[2] + (self.h // 2)), self.h)
        xx, zz = np.meshgrid(x_ax, z_ax)

        # Get the y component from plane equation
        yy = (-self.norm[0]*xx - self.norm[2]*zz - d) * 1. / self.norm[1]

        return (xx, yy, zz)


