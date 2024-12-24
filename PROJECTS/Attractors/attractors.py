import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manim_imports import *

# Ensure these constants are defined
BLUE = BLUE
WHITE = WHITE
RED = RED
GREEN = GREEN
YELLOW = YELLOW

class LorenzAttractorScene(Scene):
    pos = np.array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 24  # simulation time
    constants = (10, 28, 8 / 3)

    def construct(self):
        self.camera.frame.reorient(45, 60, 0)

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
            pts[i] = np.array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda t: pts[int(t)], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c = self.constants
        x_dot = a * (y - x)
        y_dot = x * (b - z) - y
        z_dot = x * y - c * z
        return np.array([x_dot, y_dot, z_dot])

class RosslerAttractorScene(Scene):
    pos = np.array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 24  # simulation time
    constants = (0.5, 0.2, 5.7)

    def construct(self):
        self.camera.frame.reorient(45, 60, 0)

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
            pts[i] = np.array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda t: pts[int(t)], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c = self.constants
        x_dot = -y - z
        y_dot = x + a * y
        z_dot = b + z * (x - c)
        return np.array([x_dot, y_dot, z_dot])

class AizawaAttractorScene(Scene):
    pos = np.array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 24  # simulation time
    constants = (0.95, 0.7, 0.6, 3.5, 0.25, 0.1)

    def construct(self):
        self.camera.frame.reorient(45, 60, 0)

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
            pts[i] = np.array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda t: pts[int(t)], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c, d, e, f = self.constants
        x_dot = (z - b) * x - d * y
        y_dot = d * x + (z - b) * y
        z_dot = c + a * z - (z ** 3) / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * (x ** 3)
        return np.array([x_dot, y_dot, z_dot])

class HalvorsenAttractorScene(Scene):
    pos = np.array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 24  # simulation time
    constants = (1.4,)

    def construct(self):
        self.camera.frame.reorient(45, 60, 0)

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
            pts[i] = np.array([x, y, z])

        curve = ParametricCurve(
            t_func=lambda t: pts[int(t)], t_range=[0, self.num_points - 1, 1]
        ).set_flat_stroke(False)
        return curve

    def update_curve(self, t=None, pos=None):
        x, y, z = pos
        a = self.constants[0]
        x_dot = -a * x - 4 * y - 4 * z - y**2
        y_dot = -a * y - 4 * z - 4 * x - z**2
        z_dot = -a * z - 4 * x - 4 * y - x**2
        return np.array([x_dot, y_dot, z_dot])

class CombinedAttractorScene(Scene):
    pos = np.array([0.1, 0, 0])  # initial position
    num_points = int(1e4)
    sim_time = 24  # simulation time
    lorenz_constants = (10, 28, 8 / 3)
    rossler_constants = (0.5, 0.2, 5.7)
    aizawa_constants = (0.95, 0.7, 0.6, 3.5, 0.25, 0.1)
    halvorsen_constants = (1.4,)

    def construct(self):
        self.camera.frame.reorient(0, 60, 0)

        lorenz_curve = self.get_curve(self.lorenz_constants, self.update_lorenz_curve)
        lorenz_curve.set_width(FRAME_WIDTH / 6).move_to(2 * UP + 2 * LEFT)
        lorenz_curve.set_color_by_gradient(BLUE, WHITE, BLUE)
        lorenz_axes = self.get_axes().set_width(FRAME_WIDTH / 4).move_to(2 * UP + 2 * LEFT)

        rossler_curve = self.get_curve(self.rossler_constants, self.update_rossler_curve)
        rossler_curve.set_width(FRAME_WIDTH / 6).move_to(2 * UP + 2 * RIGHT)
        rossler_curve.set_color_by_gradient(RED, WHITE, RED)
        rossler_axes = self.get_axes().set_width(FRAME_WIDTH / 4).move_to(2 * UP + 2 * RIGHT)

        aizawa_curve = self.get_curve(self.aizawa_constants, self.update_aizawa_curve)
        aizawa_curve.set_width(FRAME_WIDTH / 6).move_to(2 * DOWN + 2 * LEFT)
        aizawa_curve.set_color_by_gradient(GREEN, WHITE, GREEN)
        aizawa_axes = self.get_axes().set_width(FRAME_WIDTH / 4).move_to(2 * DOWN + 2 * LEFT)

        halvorsen_curve = self.get_curve(self.halvorsen_constants, self.update_halvorsen_curve)
        halvorsen_curve.set_width(FRAME_WIDTH / 6).move_to(2 * DOWN + 2 * RIGHT)
        halvorsen_curve.set_color_by_gradient(YELLOW, WHITE, YELLOW)
        halvorsen_axes = self.get_axes().set_width(FRAME_WIDTH / 4).move_to(2 * DOWN + 2 * RIGHT)

        # Show all axes at the same time
        self.play(ShowCreation(lorenz_axes), ShowCreation(rossler_axes), ShowCreation(aizawa_axes), ShowCreation(halvorsen_axes), run_time=self.sim_time / 8)

        # Show all curves at the same time and animate the yaw change
        self.play(
            ShowCreation(lorenz_curve),
            ShowCreation(rossler_curve),
            ShowCreation(aizawa_curve),
            ShowCreation(halvorsen_curve),
            Rotate(self.camera.frame, angle=2 * PI / (self.sim_time)**2, axis=OUT),
            # Rotate(self.camera.frame, angle=2 * PI / (self.sim_time)**2, axis=UP),
            # Rotate(self.camera.frame, angle=2 * PI / (self.sim_time)**2, axis=RIGHT),
            run_time = 2 * self.sim_time
        )

    def get_axes(self):
        axes = ThreeDAxes()
        x_label = Tex("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Tex("y").next_to(axes.y_axis.get_end(), UP)
        z_label = Tex("z").next_to(axes.z_axis.get_end(), OUT)
        axes.add(x_label, y_label, z_label)
        return axes

    def get_curve(self, constants, update_func):
            self.constants = constants
            pts = np.empty((self.num_points, 3))
            x, y, z = pts[0] = self.pos
            dt = self.sim_time / self.num_points

            for i in range(1, self.num_points):
                x_dot, y_dot, z_dot = update_func(pos=(x, y, z))
                x += x_dot * dt
                y += y_dot * dt
                z += z_dot * dt
                pts[i] = np.array([x, y, z])

            curve = ParametricCurve(
                t_func=lambda t: pts[int(t)], t_range=[0, self.num_points - 1, 1]
            ).set_flat_stroke(False)
            return curve

    def update_lorenz_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c = self.constants
        x_dot = a * (y - x)
        y_dot = x * (b - z) - y
        z_dot = x * y - c * z
        return np.array([x_dot, y_dot, z_dot])

    def update_rossler_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c = self.constants
        x_dot = -y - z
        y_dot = x + a * y
        z_dot = b + z * (x - c)
        return np.array([x_dot, y_dot, z_dot])

    def update_aizawa_curve(self, t=None, pos=None):
        x, y, z = pos
        a, b, c, d, e, f = self.constants
        x_dot = (z - b) * x - d * y
        y_dot = d * x + (z - b) * y
        z_dot = c + a * z - (z ** 3) / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * (x ** 3)
        return np.array([x_dot, y_dot, z_dot])

    def update_halvorsen_curve(self, t=None, pos=None):
        x, y, z = pos
        a = self.constants[0]
        x_dot = -a * x - 4 * y - 4 * z - y**2
        y_dot = -a * y - 4 * z - 4 * x - z**2
        z_dot = -a * z - 4 * x - 4 * y - x**2
        return np.array([x_dot, y_dot, z_dot])

# To render the scene, use the following command in the terminal:
# manimgl -pql attractors.py CombinedAttractorScene