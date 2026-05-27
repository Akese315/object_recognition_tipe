import asyncio

# Attempt to import SerialException for handling serial port errors
try:
    from scservo_sdk import SerialException
except Exception:  # Fallback if scservo_sdk is unavailable
    SerialException = Exception

import time

import matplotlib.pyplot as plt
import numpy as np

import robot


async def init_robot(rbt: robot.Robot) -> None:
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


async def plot_motor_angles_over_time(
    task: asyncio.Task,
    rbt: robot.Robot,
    time_offset,
    studied_motor_id=1,
    angles_consigne=None,
    interval: float = 0.2,
) -> None:
    start = time.time()
    timestamps: list[float] = []
    angles_per_motor: list[list[float]] = [[] for _ in range(5)]

    while not task.done():
        timestamps.append(time.time() - start)
        angles = rbt.get_arms_angle_degrees()
        for i, ang in enumerate(angles):
            angles_per_motor[i].append(ang)
        await asyncio.sleep(interval)

    # ✅ Même intervalle que la première boucle
    end_time = time.time() + time_offset
    while time.time() < end_time:
        timestamps.append(time.time() - start)
        angles = rbt.get_arms_angle_degrees()
        for i, ang in enumerate(angles):
            angles_per_motor[i].append(ang)
        await asyncio.sleep(interval)  # ✅ évite la busy-loop

    # Plot results
    fig, axs = plt.subplots(3, 2, figsize=(6, 8), sharex=True)
    axs = axs.flatten()

    for i in range(5):
        axs[i].plot(timestamps, angles_per_motor[i], label=f"Joint {i + 1}")
        if angles_consigne is not None:
            consigne_array = np.full(len(timestamps), angles_consigne[i])
            axs[i].plot(
                timestamps, consigne_array, "r--", label=f"Consigne {i + 1}"
            )  # ✅ tirets pour distinguer
        axs[i].set_ylabel("Angle (°)")
        axs[i].set_xlabel("Time (s)")
        axs[i].legend(loc="upper right")

    axs[5].axis("off")
    plt.tight_layout()
    plt.show()


async def main() -> None:
    rbt = None
    try:
        rbt = robot.Robot("COM3")
    except Exception as e:
        print("Running in simulation mode (no hardware).")
        rbt = robot.Robot("COM3")
    await init_robot(rbt)
    k_u = 255
    for i in range(len(rbt.arms)):
        rbt.arms[i].set_pid(15, 1, 0)
    await asyncio.sleep(0.1)
    initial_angles_degrees = np.array([0, 0, 20, 0, 0])
    await rbt._rotate_arms_async(initial_angles_degrees, 3400, 254)
    rbt.arms[2].set_pid(k_u, 0, 0)
    await asyncio.sleep(0.1)
    print("pid", rbt.arms[2].read_pid())
    """calculated_angles, position = rbt.IK.run_ccd(0.2, 0.2, 0.1)
    calculated_angles_degrees = calculated_angles * 360 / (2 * np.pi)
    print(calculated_angles)"""

    await asyncio.sleep(1)
    test_angles = np.array([0, 0, 0, 0, 0])
    task = asyncio.create_task(rbt._rotate_arms_async(test_angles, 3400, 254))
    await plot_motor_angles_over_time(
        task,
        rbt,
        interval=0.002,
        time_offset=5,
        angles_consigne=test_angles,
        studied_motor_id=1,
    )

    await task


if __name__ == "__main__":
    asyncio.run(main())
