import jax.numpy as jnp
from hj_reachability import dynamics
from hj_reachability import sets

speed = 0.5


class Dubins3D(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        max_turn_rate=1.0,
        control_mode="max",
        disturbance_mode="min",
        control_space=None,
        disturbance_space=None,
    ):
        self.speed = speed
        if control_space is None:
            control_space = sets.Box(
                jnp.array([-max_turn_rate]), jnp.array([max_turn_rate])
            )
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0, 0]), jnp.array([0, 0]))
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        v = self.speed
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), 0.0])

    def control_jacobian(self, state, time):
        x, y, _ = state
        return jnp.array(
            [
                [0],
                [0],
                [1],
            ]
        )

    def disturbance_jacobian(self, state, time):
        return jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )


# Define the dynamical system
dyn_sys = Dubins3D()
