import asyncio
import logging
import math
from typing import List

import numpy as np
from scservo_sdk import *
from scservo_sdk import sms_sts

from utils import clamp

# Configure basic logging
logging.basicConfig(level=logging.DEBUG)

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


def rotate_matrix(angle: np.ndarray):
    return rotate_x(angle[0]) @ rotate_y(angle[1]) @ rotate_z(angle[2])


# Deprecated: use numpy.linalg.norm directly


# Deprecated: get_argument removed; use numpy functions directly where needed


class Arm:
    def __init__(
        self, port_handler, packet_handler, servo_moteur_id, axis: str
    ) -> None:
        """Initialize an Arm instance.

        Args:
            port_handler: Handler for the serial port.
            packet_handler: Handler for packet communication.
            servo_moteur_id: Servo identifier.
            axis: Rotation axis ('x', 'y', or 'z').
        """
        self.portHandler = port_handler
        self.packetHandler = packet_handler
        self.servo_moteur_id = servo_moteur_id
        self.arm_len_vector = np.array([0, 0, 0])
        self.arm_position = np.array(self.arm_len_vector)
        self.MAX_ACCELERATION = 254
        self.MAX_STEP = 4095
        self.MAX_ROTATION_SPEED = 3400
        self.origin = 0
        self.min_angle_limit = 0
        self.max_angle_limit = 0
        self.running = False
        self.queue: list = []
        self.axis = axis

    async def rotate(
        self,
        angle: float,
        rotation_speed: float = 3400,
        acceleration_speed: float = 254,
    ) -> None:
        """Rotate the servo motor to a given angle (in degrees) at a given speed.

        Args:
            angle: Desired angle in degrees before applying origin and limits.
            rotation_speed: Target rotation speed in RPM.
            acceleration_speed: Target acceleration speed.
        """
        # Convert to physical angle respecting limits and origin
        physical_angle = self.clamp_angle(angle)
        # Convert degrees to internal step value
        logic_angle = round((physical_angle / 360) * self.MAX_STEP)
        # Convert RPM to device-specific units using constants
        logic_rotation_speed = math.floor(rotation_speed / DEG_PER_RPM)
        logic_acceleration_speed = math.floor(acceleration_speed / ACCEL_FACTOR)

        logic_rotation_speed = clamp(logic_rotation_speed, 0, self.MAX_ROTATION_SPEED)
        logic_acceleration_speed = clamp(
            logic_acceleration_speed, 0, self.MAX_ACCELERATION
        )
        self.packetHandler.WritePosEx(
            self.servo_moteur_id,
            logic_angle,
            logic_rotation_speed,
            logic_acceleration_speed,
        )

        last_position = -1
        stable_count = 0
        while True:
            position = self.read_servo_position()
            if abs(position - logic_angle) <= 5:
                break
            if last_position == position:
                stable_count += 1
                if stable_count >= 4:
                    break
            else:
                stable_count = 0
            last_position = position
            await asyncio.sleep(0.1)
        self.running = False

    def clamp_angle(self, angle):
        physical_angle = angle + self.origin
        physical_angle = clamp(
            physical_angle, self.min_angle_limit, self.max_angle_limit
        )
        return physical_angle

    def read_servo_position(self):
        state = self.packetHandler.ReadPos(self.servo_moteur_id)
        if state[1] < 0:
            logging.error("Error reading servo position")
            quit()
        return state[0]

    def get_angle_vector(self, angle: float) -> np.ndarray:
        """Return a 3‑element angle vector for the specified axis.

        Args:
            angle: Angle value in degrees.

        Raises:
            ValueError: If ``self.axis`` is not one of ``'x'``, ``'y'``, ``'z'``.
        """
        if self.axis not in ("x", "y", "z"):
            raise ValueError(
                f"Unsupported axis '{self.axis}'. Expected 'x', 'y', or 'z'."
            )
        angle_vector = np.zeros(3)
        if self.axis == "x":
            angle_vector[0] = angle
        elif self.axis == "y":
            angle_vector[1] = angle
        else:  # self.axis == "z"
            angle_vector[2] = angle
        return angle_vector

    def get_current_angle_radians(self):
        position = self.read_servo_position()
        # en radian
        position_degre = position / self.MAX_STEP * (360)
        position_relative = position_degre - self.origin
        return math.radians(position_relative)

    def read_servo_speed(self):
        state = self.packetHandler.ReadSpeed(self.servo_moteur_id)
        if state[1] < 0:
            print("Error reading servo speed")
            quit()
        return state[0]

    async def set_origin(self, origin: float) -> None:
        """Set the origin offset for the arm and reset position.

        Args:
            origin: Origin offset in degrees.
        """
        self.origin = origin
        # Await rotation to ensure servo moves to the new origin position
        await self.rotate(0)

    def set_angle_limit(self, min_angle_limit: float = 0, max_angle_limit: float = 0):
        self.min_angle_limit = min_angle_limit
        self.max_angle_limit = max_angle_limit

    def set_rotation_speed_limit(self, rotation_speed_limit: float):
        self.MAX_ROTATION_SPEED = rotation_speed_limit

    def set_acceleration_speed_limit(self, acceleration_speed_limit: float):
        self.MAX_ACCELERATION = acceleration_speed_limit

    def set_arm_len(self, arm_len_vector: np.ndarray | None = None) -> None:
        """Set the length vector of the arm.

        Args:
            arm_len_vector: Optional 3‑element array; defaults to zeros.
        """
        if arm_len_vector is None:
            arm_len_vector = np.array([0, 0, 0])
        self.arm_len_vector = arm_len_vector


class Robot:
    arms: List[Arm]

    def __init__(self, COM_PORT: str):
        self.port_handler = PortHandler(COM_PORT)
        self.packetHandler = sms_sts(self.port_handler)

        self.arms = []
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
        arm = Arm(self.port_handler, self.packetHandler, servo_moteur_id, axis)
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

    def calculate_angles_CDD(self, x_t, y_t, z_t, nb_iteration):
        """Calculate joint angles using Cyclic Coordinate Descent (CCD).

        Args:
            x_t, y_t, z_t: Target coordinates in world space.
            nb_iteration: Number of CCD iterations to perform.

        Returns:
            NumPy array of joint angles in radians.
        """
        # Current joint angles (radians) from the servos
        angles = np.array([arm.get_current_angle_radians() for arm in self.arms])
        logging.debug(f"Initial joint angles (rad): {angles}")
        N = len(self.arms)
        target = np.array([x_t, y_t, z_t])

        for _ in range(nb_iteration):
            # Iterate from the end effector back to the base
            for i in range(N):
                arm_idx = N - i - 1
                logging.debug(f"Processing arm index {arm_idx}")

                # End‑effector and current arm positions in world frame
                end_effector_pos = self.calculate_absolute_arm_position(N - 1, angles)
                arm_base_pos = (
                    self.calculate_absolute_arm_position(arm_idx - 1, angles)
                    if arm_idx > 0
                    else np.zeros(3)
                )
                logging.debug(f"End‑effector position: {end_effector_pos}")
                logging.debug(f"Arm base position: {arm_base_pos}")

                # Vectors from the current joint to the end‑effector and to the target
                e_i = end_effector_pos - arm_base_pos
                e_t = target - arm_base_pos

                # Skip update if either vector is degenerate
                if np.linalg.norm(e_i) < EPSILON or np.linalg.norm(e_t) < EPSILON:
                    logging.debug(
                        "Zero‑length vector encountered; skipping this joint."
                    )
                    continue

                # Rotation matrix that aligns the world frame to the joint's frame
                rotation_to_joint = (
                    self.get_rotation_matrix(0, arm_idx - 1, angles)
                    if arm_idx > 0
                    else np.identity(3)
                )

                # Unit vector of the motor's rotation axis expressed in world coordinates
                motor_axis_local = self.arms[arm_idx].get_angle_vector(1)
                u = rotation_to_joint @ motor_axis_local
                u_norm = np.linalg.norm(u)
                if u_norm < EPSILON:
                    logging.debug("Motor axis vector degenerate; skipping.")
                    continue
                u = u / u_norm

                # Project vectors onto the plane orthogonal to the rotation axis
                e_i_proj = e_i - np.dot(e_i, u) * u
                e_t_proj = e_t - np.dot(e_t, u) * u

                if (
                    np.linalg.norm(e_i_proj) < EPSILON
                    or np.linalg.norm(e_t_proj) < EPSILON
                ):
                    logging.debug(
                        "Projection resulted in zero‑length vector; skipping."
                    )
                    continue

                # Signed angle between the projected vectors around axis u
                dot = np.dot(e_i_proj, e_t_proj)
                cross = np.cross(e_i_proj, e_t_proj)
                sin_signed = np.dot(cross, u)
                angle_delta = math.atan2(sin_signed, dot)

                # Update joint angle (radians)
                new_angle_rad = angles[arm_idx] + angle_delta
                new_angle_deg = math.degrees(new_angle_rad)

                # Clamp the *physical* angle (includes origin) using Arm.clamp_angle
                clamped_physical_deg = self.arms[arm_idx].clamp_angle(new_angle_deg)
                # Convert back to a relative angle (subtract origin) and store in radians
                relative_deg = clamped_physical_deg - self.arms[arm_idx].origin

# End of angle update block
                angles[arm_idx] = math.radians(relative_deg)
                logging.debug(f"Arm {arm_idx}: delta={math.degrees(angle_delta):.4f}°")
                logging.debug(f"new={new_angle_deg:.4f}°, clamped={clamped_physical_deg:.4f}°")

        # Final end‑effector position for debugging (optional)
        final_pos = self.calculate_absolute_arm_position(N - 1, angles)
        logging.debug(f"Final end‑effector position: {final_pos}")
        return angles

    def calculate_absolute_arm_position(
        self, id_bras: int, angles: np.ndarray
    ) -> np.ndarray:
        rotation = np.identity(3)
        position = np.zeros(3)
        for i in range(id_bras + 1):
            angle_vector = self.arms[i].get_angle_vector(angles[i])
            rotation = rotation @ rotate_matrix(angle_vector)
            position += rotation @ self.arms[i].arm_len_vector
        return position

    def get_rotation_matrix(self, from_bras, to_bras, angles: np.ndarray):
        rotation = np.identity(3)
        for i in range(from_bras, to_bras + 1):
            angle_vector = self.arms[i].get_angle_vector(angles[i])
            rotation = rotation @ rotate_matrix(angle_vector)
        return rotation

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
            [
                0,
                0,
                z_axis_degrees,
                math.degrees(theta2_motor),
                math.degrees(theta1_motor),
            ]
        )
