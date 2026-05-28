import math

import numpy as np

from inverse_kinetic import Inverse_Kinetic


class MockArm:
    """Réplique exacte de Arm — sans appel hardware."""

    def __init__(
        self,
        arm_len_vector,
        axis="z",
        origin=0,
        min_limit=0,
        max_limit=0,
        sensitif=-1.0,
    ):
        self.servo_moteur_id = "mock"
        self.arm_len_vector = np.asarray(arm_len_vector, dtype=float)
        self.origin = origin
        self.min_angle_limit = min_limit
        self.max_angle_limit = max_limit
        self.sens_rotation = sensitif
        self.axis = axis
        self.current_angle_rad = 0.0

    def get_angle_vector(self, angle):
        """Simule Arm.get_angle_vector — l'angle est sur l'axe du bras."""
        vec = np.zeros(3)
        if self.axis == "x":
            vec[0] = angle
        elif self.axis == "y":
            vec[1] = angle
        elif self.axis == "z":
            vec[2] = angle
        return vec

    def get_current_angle_radians(self):
        return self.current_angle_rad

    def clamp_angle(self, angle_deg):
        """Même logique que Arm.clamp_angle."""
        physical = angle_deg * self.sens_rotation + self.origin
        return np.clip(physical, self.min_angle_limit, self.max_angle_limit)

    def set_origin(self, origin):
        self.origin = origin

    def set_angle_limit(self, min_lim, max_lim):
        self.min_angle_limit = min_lim
        self.max_angle_limit = max_lim

    def set_arm_len(self, arm_len):
        self.arm_len_vector = np.asarray(arm_len, dtype=float)

    def rotate(self, angle_deg):
        self.current_angle_rad = math.radians(angle_deg)


# ============================================================================


def run_test(test_name, arms, target):
    """Exécute un test CCD et affiche les résultats."""
    x_t, y_t, z_t = target
    ik = Inverse_Kinetic(arms, 50, debug=False)

    print(f"\n{'=' * 60}")
    print(f"TEST : {test_name}")
    print(f"Cible : ({x_t:.4f}, {y_t:.4f}, {z_t:.4f}) m")
    print(f"{'=' * 60}")

    if not ik.is_reachable(x_t, y_t, z_t):
        print(">>> Cible inatteignable\n")
        return

    angles, effector_pos = ik.run_ccd(x_t, y_t, z_t)
    error = np.linalg.norm(np.array([x_t, y_t, z_t]) - effector_pos)

    print(f"Angles finaux (rad)  : {angles}")
    print(f"Angles finaux (deg)  : {np.degrees(angles)}")
    print(f"Effecteur final      : {effector_pos}")
    print(f"Erreur euclidienne   : {error:.6f} m  ({error * 100:.4f} cm)")


if __name__ == "__main__":
    # --- Configuration des bras ---
    mock_arms = [
        MockArm(
            np.array([0.035, 0.115, 0]),
            axis="y",
            origin=180,
            min_limit=90,
            max_limit=270,
        ),
        MockArm(
            np.array([0.03, 0.115, 0]),
            axis="z",
            origin=180,
            min_limit=90,
            max_limit=270,
        ),
        MockArm(
            np.array([0, 0.135, 0]), axis="z", origin=90, min_limit=90, max_limit=270
        ),
        MockArm(
            np.array([0, 0.06, 0]), axis="z", origin=180, min_limit=90, max_limit=270
        ),
        MockArm(np.array([0, 0.1, 0]), axis="y", origin=45, min_limit=0, max_limit=180),
    ]

    # Initialiser les angles comme le ferait un robot vrai après set_origin
    mock_arms[0].current_angle_rad = 0.0
    mock_arms[1].current_angle_rad = 0.0
    mock_arms[2].current_angle_rad = 0.0
    mock_arms[3].current_angle_rad = 0.0
    mock_arms[4].current_angle_rad = -math.pi / 2.0  # origine 45

    """test_cases = [
        ("Cible éloignée", (0.0, 0.0, 0.30)),
        ("Cible décalée", (0.10, 0.10, 0.30)),
        ("Cible courte", (0.0, 0.10, 0.20)),
        ("Inatteignable (trop loin)", (0.0, 0.0, 0.60)),
    ]"""

    test_cases = [("cas spécial", (0.15, 0.1, 0.1))]

    for name, target in test_cases:
        # Reset des angles avant chaque test
        for arm in mock_arms:
            arm.current_angle_rad = 0.0
        run_test(name, mock_arms, target)

    print("\n=== Tests terminés ===")
