import asyncio
import logging
import math
from typing import List

import numpy as np
from scservo_sdk import *
from scservo_sdk import sms_sts

from arm import Arm
from inverse_kinetic import Inverse_Kinetic

# Configure basic logging
logging.basicConfig(level=logging.INFO)
# Suppress verbose websockets debug logs
logging.getLogger("websockets").setLevel(logging.WARNING)

# Constants
EPSILON = 1e-7  # Tolerance for zero-length vectors
DEG_PER_RPM = 0.732  # Conversion factor from RPM to internal units
ACCEL_FACTOR = 8.7  # Conversion factor for acceleration


def rotate_y(angle):
    """Créer une matrice rotation autour de l'axe y d'un angle en radian"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]
    )


def rotate_x(angle):
    """Créer une matrice rotation autour de l'axe x d'un angle en radian"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )


def rotate_z(angle):
    """Créer une matrice rotation autour de l'axe z d'un angle en radian"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def rotation_for_axis(axis: str, angle: float) -> np.ndarray:
    """Return a rotation matrix for a single axis.

    Args:
        axis: Axis identifier ('x', 'y', or 'z').
        angle: Rotation angle in radians.
    """
    if axis == "x":
        return rotate_x(angle)
    if axis == "y":
        return rotate_y(angle)
    if axis == "z":
        return rotate_z(angle)
    raise ValueError(f"Unsupported axis '{axis}'. Expected 'x', 'y', or 'z'.")


def rotate_matrix(angle: np.ndarray):
    """Compose rotation matrix from Euler angles.
    The order Z → Y → X matches the convention used in the CCD reference
    (first rotate around X, then Y, then Z in the local frame, which corresponds
    to a matrix multiplication Z·Y·X when expressed in the world frame)."""
    return rotate_z(angle[2]) @ rotate_y(angle[1]) @ rotate_x(angle[0])


# Deprecated: use numpy.linalg.norm directly


# Deprecated: get_argument removed; use numpy functions directly where needed


class Robot:
    arms: List[Arm]

    def __init__(self, COM_PORT: str):
        self.port_handler = PortHandler(COM_PORT)
        self.packetHandler = sms_sts(self.port_handler)
        self.arms = []
        self.IK = Inverse_Kinetic(self.arms, 20)
        if self.port_handler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()
        if self.port_handler.setBaudRate(1000000):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

    async def add_arm(
        self,
        servo_moteur_id: int,
        origin: float,
        axis: str,
        min_angle_limit: float = 0,
        max_angle_limit: float = 0,
        sens_rotation: float = 1,
        arm_len: np.ndarray | None = None,
    ) -> None:
        """Create an Arm, configure it, and add to the robot.

        Args:
            servo_moteur_id: Identifier of the servo motor.
            origin: Origin offset in degrees.
            axis: Rotation axis for the arm ('x', 'y', 'z').
            min_angle_limit: Minimum allowed angle (degrees).
            max_angle_limit: Maximum allowed angle (degrees).
            arm_len: Length vector; defaults to zero vector if None.
        """
        arm = Arm(
            self.port_handler, self.packetHandler, servo_moteur_id, sens_rotation, axis
        )
        arm.set_angle_limit(min_angle_limit, max_angle_limit)
        await arm.set_origin(origin)
        arm.set_arm_len(arm_len)
        self.arms.append(arm)

    def __del__(self):
        self.port_handler.closePort()

    async def _rotate_arms_async(
        self,
        angles: np.ndarray,
        rotation_speed: float,
        acceleration_speed: float,
    ):
        tasks = []
        print("rotate")
        for i in range(len(self.arms)):
            tasks.append(
                self.arms[i].rotate(angles[i], rotation_speed, acceleration_speed)
            )
        await asyncio.gather(*tasks)

    def rotate_arms(
        self,
        angles: np.ndarray,
        rotation_speed: float = 3400,
        acceleration_speed: float = 254,
    ):
        asyncio.run(self._rotate_arms_async(angles, rotation_speed, acceleration_speed))

    def get_arms_angle_degrees(self):
        angles = []
        for arm in self.arms:
            angle = math.degrees(arm.get_current_angle_radians())
            angles.append(angle)
        return angles

    def move_to_2arms(self, x_t, y_t, z_axis_degrees):
        # Longueurs des bras définies dans votre code
        L1 = 0.12
        L2 = 0.31
        r_sq = x_t**2 + y_t**2
        r = math.sqrt(r_sq)
        if math.fabs(L2 - L1) > r or r > (L1 + L2):
            print("Distance too big or too small")
            return
        C = (L2**2 - L1**2 - r_sq) / (-2 * L1)
        theta1_geo = math.atan2(y_t, x_t) + math.acos(C / r)
        const_theta_2 = (L1**2 - L2**2 - (x_t**2 + y_t**2)) / (-2 * L2)
        theta2_geo = -math.acos(const_theta_2 / r) + math.atan2(y_t, x_t) - theta1_geo
        theta1_motor = math.pi - theta1_geo
        theta2_motor = -theta2_geo
        print(
            f"Angles Géométriques: th1={math.degrees(theta1_geo):.2f}°, th2={math.degrees(theta2_geo):.2f}°"
        )
        print(
            f"Angles Moteurs: th1={math.degrees(theta1_motor):.2f}°, th2={math.degrees(theta2_motor):.2f}°"
        )
        self.rotate_arms(
            np.array(
                [
                    0,
                    0,
                    z_axis_degrees,
                    math.degrees(theta2_motor),
                    math.degrees(theta1_motor),
                ]
            )
        )
