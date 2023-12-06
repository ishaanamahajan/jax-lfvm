import jax.numpy as jnp
from jax import grad, jit
from collections import namedtuple

# Define the vehicle's state and parameters
VehStates = namedtuple('VehStates', ['x', 'y', 'theta', 'v'])
VehParam = namedtuple('VehParam', ['step', 'l', 'R', 'gamma', 'tau0', 'omega0', 'c1', 'c0', 'I'])

def integrate_any(v_st, v_params, controls):
    x_update = v_st.x + jnp.cos(v_st.theta) * v_st.v * v_params.step
    y_update = v_st.y + jnp.sin(v_st.theta) * v_st.v * v_params.step
    theta_update = v_st.theta + (jnp.tan(v_params.steerMap * controls[1]) / v_params.l) * v_st.v * v_params.step

    rGamma = v_params.R * v_params.gamma
    f1 = v_params.tau0 * controls[2] - (v_params.tau0 * v_st.v / (v_params.omega0 * rGamma))
    f0 = v_st.v * v_params.c1 / rGamma + v_params.c0
    ta = f1 - f0
    ta = jnp.where(jnp.abs(v_st.v) < 1e-9, jnp.maximum(ta, 0), ta)

    v_update = v_st.v + ((rGamma / v_params.I) * ta) * v_params.step

    return VehStates(x_update, y_update, theta_update, v_update)

# Example usage
v_st = VehStates(x=0, y=0, theta=0, v=0)
v_params = VehParam(step=0.01, l=2.5, R=0.3, gamma=1.0, tau0=100, omega0=100, c1=0.5, c0=0.1, I=1500)
controls = [0, 0.1, 1.0]  # Example control inputs

# Integrate vehicle states
new_v_st = integrate_any(v_st, v_params, controls)

