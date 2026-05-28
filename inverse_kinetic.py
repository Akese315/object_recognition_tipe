import math

import numpy as np

from arm import Arm


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


def rotate(angle_vector: np.ndarray):
    """Retourne la bonne matrice de rotation. Puisque le moteur ne peut qu'avoir une composante, 2 matrices sur 3 deviennent la matrice identité, donc pas de problème d'ordre"""
    return (
        rotate_x(angle_vector[0])
        @ rotate_y(angle_vector[1])
        @ rotate_z(angle_vector[2])
    )


class Inverse_Kinetic:
    def __init__(self, arms: list[Arm], nb_iteration, debug: bool = False):
        self.arms = arms
        self.nb_iteration = nb_iteration
        self.debug = debug

        # Désactive le format scientifique numpy
        np.set_printoptions(suppress=True)

    def dprint(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    # =========================
    # FORMAT DEBUG HUMAIN
    # =========================

    def fmt(self, value):
        """
        Affichage lisible :
        - scalaire -> 2 décimales
        - tableau numpy -> 2 décimales
        """

        if isinstance(value, np.ndarray):
            return np.array2string(
                value,
                precision=2,
                suppress_small=True,
                floatmode="fixed",
            )

        if isinstance(value, (float, np.floating)):
            return f"{value:.2f}"

        return value

    def get_tip_position(self, motor_id, angles: np.ndarray):
        """Renvoie la position du moteur dans la base canonique R^3"""

        position = np.zeros(3)
        rotation_matrix = np.eye(3)

        index = 0

        while index <= motor_id:
            self.dprint(
                f"-------------get_tip_position EFFECTOR index : {index}---------------"
            )

            angle_vector = self.arms[index].get_angle_vector(angles[index])

            self.dprint(f"arm_id : {self.arms[index].servo_moteur_id}")
            self.dprint(f"angle_vector : {self.fmt(angle_vector)}")

            rotation_matrix = rotation_matrix @ rotate(angle_vector)

            position += rotation_matrix @ self.arms[index].arm_len_vector

            self.dprint(f"position : {self.fmt(position)}")

            index += 1

        return position

    def get_rotation_matrix(self, motor_id, angles: np.ndarray):

        index = 0
        rotation_matrix = np.eye(3)

        while index <= motor_id:
            angle_vector = self.arms[index].get_angle_vector(angles[index])

            rotation_matrix = rotation_matrix @ rotate(angle_vector)

            index += 1

        return rotation_matrix

    def is_reachable(self, x_t: float, y_t: float, z_t: float) -> bool:
        """
        Vérifie si la position cible se trouve dans le rayon d'action maximal du robot.
        Retourne True si la cible est potentiellement atteignable, False sinon.
        """
        target_position = np.array([x_t, y_t, z_t])

        distance_to_target = np.linalg.norm(target_position)

        max_reach = 0.0
        for arm in self.arms:
            max_reach += np.linalg.norm(arm.arm_len_vector)

        epsilon = 1e-5

        if distance_to_target > (max_reach + epsilon):
            self.dprint(
                f"⚠️ Cible INATTEIGNABLE : La distance ({distance_to_target:.4f}) dépasse l'allonge maximale ({max_reach:.4f})."
            )
            return False

        return True

    def run_ccd(self, x_t: float, y_t: float, z_t: float):

        N = len(self.arms)

        target_position = np.array([x_t, y_t, z_t])

        angles = np.array([self.arms[i].get_current_angle_radians() for i in range(N)])

        end_effector_position = self.get_tip_position(N - 1, angles)

        self.dprint("\n========== CCD START ==========")
        self.dprint(f"Target position : {self.fmt(target_position)}")
        self.dprint(f"Initial angles  : {self.fmt(angles)}")
        self.dprint(f"Initial effector: {self.fmt(end_effector_position)}")

        if not self.is_reachable(x_t, y_t, z_t):
            print("Erreur, la position n'est pas atteignable", end="\r", flush=True)
            return np.zeros(len(self.arms)), np.zeros(3)

        for iteration in range(self.nb_iteration):
            self.dprint(f"\n\n===== ITERATION {iteration} =====")

            for i in range(1, N):
                effector_index = N - i - 1

                self.dprint(f"\n--- Arm {effector_index + 1} ---")

                end_effector_position = self.get_tip_position(N - 1, angles)

                index_effector_position = self.get_tip_position(
                    effector_index - 1,
                    angles,
                )

                self.dprint(f"Current angles                : {self.fmt(angles)}")
                self.dprint(
                    f"Joint position                : {self.fmt(index_effector_position)}"
                )
                self.dprint(
                    f"End effector position         : {self.fmt(end_effector_position)}"
                )

                e_i = end_effector_position - index_effector_position
                e_t = target_position - index_effector_position

                self.dprint(f"e_i (joint -> effector)       : {self.fmt(e_i)}")
                self.dprint(f"e_t (joint -> target)         : {self.fmt(e_t)}")

                angle_vector_unitaire = self.arms[effector_index].get_angle_vector(1)

                self.dprint(
                    f"Rotation axis local           : {self.fmt(angle_vector_unitaire)}"
                )

                rotation_matrix = self.get_rotation_matrix(
                    effector_index,
                    angles,
                )

                self.dprint(f"Rotation matrix:\n{self.fmt(rotation_matrix)}")

                u = rotation_matrix @ angle_vector_unitaire

                self.dprint(f"Rotation axis world (u)       : {self.fmt(u)}")

                e_i_proj = e_i - np.dot(e_i, u) * u
                e_t_proj = e_t - np.dot(e_t, u) * u

                self.dprint(f"e_i_proj                      : {self.fmt(e_i_proj)}")
                self.dprint(f"e_t_proj                      : {self.fmt(e_t_proj)}")

                norme_e_i_proj = np.linalg.norm(e_i_proj)
                norme_e_t_proj = np.linalg.norm(e_t_proj)

                self.dprint(
                    f"|e_i_proj|                    : {self.fmt(norme_e_i_proj)}"
                )
                self.dprint(
                    f"|e_t_proj|                    : {self.fmt(norme_e_t_proj)}"
                )

                produit_vectoriel = np.cross(e_i_proj, e_t_proj)

                produit_vectoriel_signe = np.dot(
                    u,
                    produit_vectoriel,
                )

                self.dprint(
                    f"Cross product                 : {self.fmt(produit_vectoriel)}"
                )
                self.dprint(
                    f"Signed cross product          : {self.fmt(produit_vectoriel_signe)}"
                )

                alpha_proj = np.arctan2(
                    produit_vectoriel_signe,
                    np.dot(e_t_proj, e_i_proj),
                )

                self.dprint(f"alpha_proj (rad)              : {self.fmt(alpha_proj)}")
                self.dprint(
                    f"alpha_proj (deg)              : {self.fmt(math.degrees(alpha_proj))}"
                )

                angle_papier = angles[effector_index] + alpha_proj

                self.dprint(f"Raw new angle (rad)           : {self.fmt(angle_papier)}")
                self.dprint(
                    f"Raw new angle (deg)           : {self.fmt(math.degrees(angle_papier))}"
                )

                clamped_deg = self.arms[effector_index].clamp_angle(
                    math.degrees(angle_papier)
                )

                self.dprint(f"Clamped angle (deg)           : {self.fmt(clamped_deg)}")

                angle_papier_clamped = math.radians(
                    (clamped_deg - self.arms[effector_index].origin)
                    / self.arms[effector_index].sens_rotation
                )

                self.dprint(
                    f"Final stored angle (rad)      : {self.fmt(angle_papier_clamped)}"
                )

                angles[effector_index] = angle_papier_clamped

                self.dprint(f"Updated angles (rad)          : {self.fmt(angles)}")
                self.dprint(
                    f"Updated angles (deg)          : {self.fmt(angles * 360 / (2 * np.pi))}"
                )

            final_effector = self.get_tip_position(
                N - 1,
                angles,
            )

            error = np.linalg.norm(target_position - final_effector)

            self.dprint(f"\nIteration {iteration} summary")
            self.dprint(f"Effector position             : {self.fmt(final_effector)}")
            self.dprint(f"Distance to target            : {self.fmt(error)}")

        self.dprint("\n========== CCD END ==========")

        return angles, end_effector_position
