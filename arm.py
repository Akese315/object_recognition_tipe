import asyncio
import logging
import math

import numpy as np

from utils import clamp

# Configure basic logging
logging.basicConfig(level=logging.INFO)
# Suppress verbose websockets debug logs
logging.getLogger("websockets").setLevel(logging.WARNING)

# Constants
EPSILON = 1e-7  # Tolerance for zero-length vectors
DEG_PER_RPM = 0.732  # Conversion factor from RPM to internal units
ACCEL_FACTOR = 8.7  # Conversion factor for acceleration


class Arm:
    def __init__(
        self, port_handler, packet_handler, servo_moteur_id, sens_rotation, axis: str
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
        self.sens_rotation = sens_rotation

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
        physical_angle = self.sens_rotation * angle + self.origin
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
        elif self.axis == "z":
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
