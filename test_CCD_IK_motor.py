"""
Test script pour l'inverse kinematics (algorithme CCD).
Version 100 % logicielle — aucun hardware requis.
"""

import asyncio
import math

import numpy as np

from robot import Robot


async def init_robot(rbt: Robot) -> None:
    await rbt.add_arm(
        servo_moteur_id=1,
        origin=180,
        axis="y",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0.035, 0.115, 0]),
    )

    await rbt.add_arm(
        servo_moteur_id=2,
        origin=180,
        axis="z",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0.03, 0.115, 0]),
    )
    await rbt.add_arm(
        servo_moteur_id=3,
        origin=90,
        axis="z",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.135, 0]),
    )

    await rbt.add_arm(
        servo_moteur_id=4,
        origin=180,
        axis="z",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.06, 0]),
    )
    await rbt.add_arm(
        servo_moteur_id=5,
        origin=45,
        axis="y",
        min_angle_limit=0,
        max_angle_limit=180,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.1, 0]),
    )

    await rbt.add_pince(
        servo_moteur_id=6,
        origin=180,
        axis="y",
        min_angle_limit=90,
        max_angle_limit=180,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.1, 0]),
    )


async def main():
    rbt = Robot("COM3")
    await init_robot(rbt)

    # --- Définition des tests ---
    """test_cases = [
        ("Cible éloignée", (0.0, 0.0, 0.30)),
        ("Cible décalée", (0.10, 0.10, 0.30)),
        ("Cible courte", (0.0, 0.10, 0.20)),
        ("Inatteignable (trop loin)", (0.0, 0.0, 0.60)),
    ]"""

    test_cases = [("cas spécial", (0.15, 0.1, 0.1))]

    for name, target in test_cases:
        # Reset des angles avant chaque test
        x_t = target[0]
        y_t = target[1]
        z_t = target[2]
        angles, effector_pos = rbt.IK.run_ccd(x_t, y_t, z_t)
        angles_degrees = angles * 180 / (np.pi)
        error = np.linalg.norm(np.array([x_t, y_t, z_t]) - effector_pos)

        print(f"Angles finaux (rad)  : {angles}")
        print(f"Angles finaux (deg)  : {np.degrees(angles)}")
        print(f"Effecteur final      : {effector_pos}")
        print(f"Erreur euclidienne   : {error:.6f} m  ({error * 100:.4f} cm)")
        await rbt._rotate_arms_async(angles_degrees, 2000, 200)

    print("\n=== Tests terminés ===")


if __name__ == "__main__":
    asyncio.run(main())
