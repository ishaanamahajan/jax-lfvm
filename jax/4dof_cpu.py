import jax.numpy as jnp
from jax import jit, vmap
import time
from collections import namedtuple

# Define the vehicle's state and parameters using namedtuples
VehStates = namedtuple('VehStates', ['x', 'y', 'theta', 'v'])
VehParam = namedtuple('VehParam', ['step', 'l', 'R', 'gamma', 'tau0', 'omega0', 'c1', 'c0', 'I', 'steerMap'])


def veh_states_to_array(v_st):
    return jnp.array([v_st.x, v_st.y, v_st.theta, v_st.v])

def array_to_veh_states(arr):
    return VehStates(x=arr[0], y=arr[1], theta=arr[2], v=arr[3])


@jit
def integrate_any(v_st_array, v_params, controls):
    x, y, theta, v = v_st_array
    x_update = x + jnp.cos(theta) * v * v_params.step
    y_update = y + jnp.sin(theta) * v * v_params.step
    theta_update = theta + (jnp.tan(v_params.steerMap * controls[1]) / v_params.l) * v * v_params.step

    rGamma = v_params.R * v_params.gamma
    f1 = v_params.tau0 * controls[2] - (v_params.tau0 * v / (v_params.omega0 * rGamma))
    f0 = v * v_params.c1 / rGamma + v_params.c0
    ta = f1 - f0
    ta = jnp.where((jnp.abs(v - 0) < 1e-9) & (ta < 0), 0, ta)

    v_update = v + ((rGamma / v_params.I) * ta) * v_params.step

    return jnp.array([x_update, y_update, theta_update, v_update])


vectorized_integrate_any = vmap(integrate_any, in_axes=(0, None, 0))

# Main function to run the parallel simulations
def main():
    num_simulations = 10  
    num_steps = 10  

    # Vehicle parameters (assuming the same for all simulations)
    v_params = VehParam(step=0.001, l=2.5, R=0.3, gamma=1.0, tau0=100, omega0=100, c1=0.5, c0=0.1, I=1500, steerMap=1.0)
    initial_states = jnp.array([veh_states_to_array(VehStates(x=0, y=0, theta=0, v=i)) for i in range(num_simulations)])

    all_controls = jnp.zeros((num_simulations, num_steps, 3))  

    
    start_time = time.time()

    
    states = initial_states
    for step in range(num_steps):
        states = vectorized_integrate_any(states, v_params, all_controls[:, step, :])

    
    end_time = time.time()

    
    duration_sec = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Parallel Simulation Duration: {duration_sec} ms")

    
    final_states = [array_to_veh_states(state) for state in states]
    #print(final_states)

if __name__ == "__main__":
    main()
