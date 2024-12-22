import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manim_imports import *
from numpy import sin, cos, array
from scipy.integrate import solve_ivp


class LorenzAttractorScene(Scene):
    pos = array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 50  # simulation time
    constants = (10, 28, 8 / 3)

    def construct(self):
        self.frame.reorient(45, 60, 0)

        lorenz_curve = self.get_lorenz_curve()
        lorenz_curve.set_width(FRAME_WIDTH / 2.5).center()
        lorenz_curve.set_color_by_gradient(BLUE, WHITE, BLUE)
        self.play(ShowCreation(lorenz_curve), run_time=self.sim_time)

    def get_lorenz_curve(self):
        pts = np.empty((self.num_points, 3))
        x, y, z = pts[0] = self.pos
        dt = self.sim_time / self.num_points

        for i in range(1, self.num_points):
            x_dot, y_dot, z_dot = self.update_curve(pos=(x, y, z))
            x += x_dot * dt
            y += y_dot * dt
            z += z_dot * dt
            pts[i] = array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda i: pts[i], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c = self.constants
        x_dot = a * (y - x)
        y_dot = x * (b - z) - y
        z_dot = x * y - c * z
        return array([x_dot, y_dot, z_dot])


class RosslerAttractorScene(Scene):
    pos = array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 50  # simulation time
    constants = (0.2, 0.2, 5.7)

    def construct(self):
        self.frame.reorient(45, 60, 0)

        rossler_curve = self.get_rossler_curve()
        rossler_curve.set_width(FRAME_WIDTH / 2.5).center()
        rossler_curve.set_color_by_gradient(RED, WHITE, RED)
        self.play(ShowCreation(rossler_curve), run_time=self.sim_time)

    def get_rossler_curve(self):
        pts = np.empty((self.num_points, 3))
        x, y, z = pts[0] = self.pos
        dt = self.sim_time / self.num_points

        for i in range(1, self.num_points):
            x_dot, y_dot, z_dot = self.update_curve(pos=(x, y, z))
            x += x_dot * dt
            y += y_dot * dt
            z += z_dot * dt
            pts[i] = array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda i: pts[i], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c = self.constants
        x_dot = -y - z
        y_dot = x + a * y
        z_dot = b + z * (x - c)
        return array([x_dot, y_dot, z_dot])
    

class AizawaAttractorScene(Scene):
    pos = array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 50  # simulation time
    constants = (0.95, 0.7, 0.6, 3.5, 0.25, 0.1)

    def construct(self):
        self.frame.reorient(45, 60, 0)

        aizawa_curve = self.get_aizawa_curve()
        aizawa_curve.set_width(FRAME_WIDTH / 2.5).center()
        aizawa_curve.set_color_by_gradient(GREEN, WHITE, GREEN)
        self.play(ShowCreation(aizawa_curve), run_time=self.sim_time)

    def get_aizawa_curve(self):
        pts = np.empty((self.num_points, 3))
        x, y, z = pts[0] = self.pos
        dt = self.sim_time / self.num_points

        for i in range(1, self.num_points):
            x_dot, y_dot, z_dot = self.update_curve(pos=(x, y, z))
            x += x_dot * dt
            y += y_dot * dt
            z += z_dot * dt
            pts[i] = array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda i: pts[i], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c, d, e, f = self.constants
        x_dot = (z - b) * x - d * y
        y_dot = d * x + (z - b) * y
        z_dot = c + a * z - (z ** 3) / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * (x ** 3)
        return array([x_dot, y_dot, z_dot])
    

class HalvorsenAttractorScene(Scene):
    pos = array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 50  # simulation time
    constants = (1.4,)

    def construct(self):
        self.frame.reorient(45, 60, 0)

        halvorsen_curve = self.get_halvorsen_curve()
        halvorsen_curve.set_width(FRAME_WIDTH / 2.5).center()
        halvorsen_curve.set_color_by_gradient(YELLOW, WHITE, YELLOW)
        self.play(ShowCreation(halvorsen_curve), run_time=self.sim_time)

    def get_halvorsen_curve(self):
        pts = np.empty((self.num_points, 3))
        x, y, z = pts[0] = self.pos
        dt = self.sim_time / self.num_points

        for i in range(1, self.num_points):
            x_dot, y_dot, z_dot = self.update_curve(pos=(x, y, z))
            x += x_dot * dt
            y += y_dot * dt
            z += z_dot * dt
            pts[i] = array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda i: pts[i], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a = self.constants[0]
        x_dot = -a * x - 4 * y - 4 * z - y**2
        y_dot = -a * y - 4 * z - 4 * x - z**2
        z_dot = -a * z - 4 * x - 4 * y - x**2
        return array([x_dot, y_dot, z_dot])
    

def get_lorenz_curve(pos, num_points, sim_time, constants):
    pts = np.empty((num_points, 3))
    x, y, z = pts[0] = pos
    dt = sim_time / num_points

    for i in range(1, num_points):
        x_dot, y_dot, z_dot = update_lorenz_curve((x, y, z), constants)
        x += x_dot * dt
        y += y_dot * dt
        z += z_dot * dt
        pts[i] = array([x, y, z])

    curve = ParametricCurve(
        t_func=lambda i: pts[i], t_range=[0, num_points - 1, 1]
    ).set_flat_stroke(False)
    return curve
def update_lorenz_curve(pos, constants):
    x, y, z = pos
    a, b, c = constants
    x_dot = a * (y - x)
    y_dot = x * (b - z) - y
    z_dot = x * y - c * z
    return array([x_dot, y_dot, z_dot])

def get_rossler_curve(pos, num_points, sim_time, constants):
    pts = np.empty((num_points, 3))
    x, y, z = pts[0] = pos
    dt = sim_time / num_points

    for i in range(1, num_points):
        x_dot, y_dot, z_dot = update_rossler_curve((x, y, z), constants)
        x += x_dot * dt
        y += y_dot * dt
        z += z_dot * dt
        pts[i] = array([x, y, z])

    curve = ParametricCurve(
        t_func=lambda i: pts[i], t_range=[0, num_points - 1, 1]
    ).set_flat_stroke(False)
    return curve
def update_rossler_curve(pos, constants):
    x, y, z = pos
    a, b, c = constants
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return array([x_dot, y_dot, z_dot])

def get_aizawa_curve(pos, num_points, sim_time, constants):
    pts = np.empty((num_points, 3))
    x, y, z = pts[0] = pos
    dt = sim_time / num_points

    for i in range(1, num_points):
        x_dot, y_dot, z_dot = update_aizawa_curve((x, y, z), constants)
        x += x_dot * dt
        y += y_dot * dt
        z += z_dot * dt
        pts[i] = array([x, y, z])

    curve = ParametricCurve(
        t_func=lambda i: pts[i], t_range=[0, num_points - 1, 1]
    ).set_flat_stroke(False)
    return curve
def update_aizawa_curve(pos, constants):
    x, y, z = pos
    a, b, c, d, e, f = constants
    x_dot = (z - b) * x - d * y
    y_dot = d * x + (z - b) * y
    z_dot = c + a * z - (z ** 3) / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * (x ** 3)
    return array([x_dot, y_dot, z_dot])

def get_halvorsen_curve(pos, num_points, sim_time, constants):
    pts = np.empty((num_points, 3))
    x, y, z = pts[0] = pos
    dt = sim_time / num_points

    for i in range(1, num_points):
        x_dot, y_dot, z_dot = update_halvorsen_curve((x, y, z), constants)
        x += x_dot * dt
        y += y_dot * dt
        z += z_dot * dt
        pts[i] = array([x, y, z])

    curve = ParametricCurve(
        t_func=lambda i: pts[i], t_range=[0, num_points - 1, 1]
    ).set_flat_stroke(False)
    return curve
def update_halvorsen_curve(pos, constants):
    x, y, z = pos
    a = constants[0]
    x_dot = -a * x - 4 * y - 4 * z - y**2
    y_dot = -a * y - 4 * z - 4 * x - z**2
    z_dot = -a * z - 4 * x - 4 * y - x**2
    return array([x_dot, y_dot, z_dot])

class CombinedAttractorScene(Scene):
    # Parameters for the attractors
    pos = array([0.1, 0, 0])
    num_points = int(1e4)
    sim_time = 50
    constants_lorenz = (10, 28, 8 / 3)
    constants_rossler = (0.2, 0.2, 5.7)
    constants_aizawa = (0.95, 0.7, 0.6, 3.5, 0.25, 0.1)
    constants_halvorsen = (1.4,)

    def construct(self):
        self.frame.reorient(45, 60, 0)

        # Get the curves
        lorenz_curve = get_lorenz_curve(self.pos, self.num_points, self.sim_time, self.constants_lorenz)
        rossler_curve = get_rossler_curve(self.pos, self.num_points, self.sim_time, self.constants_rossler)
        aizawa_curve = get_aizawa_curve(self.pos, self.num_points, self.sim_time, self.constants_aizawa)
        halvorsen_curve = get_halvorsen_curve(self.pos, self.num_points, self.sim_time, self.constants_halvorsen)

        # Scale, center and color the curves
        lorenz_curve.set_width(FRAME_WIDTH / 2.5).center()
        lorenz_curve.set_color_by_gradient(BLUE, WHITE, BLUE)
        rossler_curve.set_width(FRAME_WIDTH / 2.5).center()
        rossler_curve.set_color_by_gradient(RED, WHITE, RED)
        aizawa_curve.set_width(FRAME_WIDTH / 2.5).center()
        aizawa_curve.set_color_by_gradient(GREEN, WHITE, GREEN)
        halvorsen_curve.set_width(FRAME_WIDTH / 2.5).center()
        halvorsen_curve.set_color_by_gradient(YELLOW, WHITE, YELLOW)

        # Add the curves to the scene
        self.add(lorenz_curve, rossler_curve, aizawa_curve, halvorsen_curve)

        # Play the creation of the curves
        self.play(
            ShowCreation(lorenz_curve),
            ShowCreation(rossler_curve),
            ShowCreation(aizawa_curve),
            ShowCreation(halvorsen_curve),
            run_time=self.sim_time
        )

# To render the scene, use the following command in the terminal:
# manimgl -pql attractors.py CombinedAttractorScene