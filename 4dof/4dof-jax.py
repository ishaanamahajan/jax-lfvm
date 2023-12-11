import jax.numpy as jnp
from jax import jit
from collections import namedtuple
import time

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# Define the vehicle's state and parameters using namedtuples
VehStates = namedtuple('VehStates', ['x', 'y', 'theta', 'v'])
VehParam = namedtuple('VehParam', ['step', 'l', 'R', 'gamma', 'tau0', 'omega0', 'c1', 'c0', 'I', 'steerMap'])
Entry = namedtuple('Entry', ['m_time', 'm_steering', 'm_throttle', 'm_braking'])

# Function to read driver input from a file
def driver_input(m_data, filename):
    with open(filename, 'r') as ifile:
        for line in ifile:
            parts = line.split()
            if len(parts) != 4:
                continue
            time, steering, throttle, braking = map(float, parts)
            m_data.append(Entry(time, steering, throttle, braking))

# Function to get controls based on the current time
def get_controls(controls, m_data, time):
    if time <= m_data[0].m_time:
        controls[0] = time
        controls[1] = m_data[0].m_steering
        controls[2] = m_data[0].m_throttle
        controls[3] = m_data[0].m_braking
    elif time >= m_data[-1].m_time:
        controls[0] = time
        controls[1] = m_data[-1].m_steering
        controls[2] = m_data[-1].m_throttle
        controls[3] = m_data[-1].m_braking

# Function to integrate vehicle states
@jit
def integrate_any(v_st, v_params, controls):
    x_update = v_st.x + jnp.cos(v_st.theta) * v_st.v * v_params.step
    y_update = v_st.y + jnp.sin(v_st.theta) * v_st.v * v_params.step
    theta_update = v_st.theta + (jnp.tan(v_params.steerMap * controls[1]) / v_params.l) * v_st.v * v_params.step

    rGamma = v_params.R * v_params.gamma
    f1 = v_params.tau0 * controls[2] - (v_params.tau0 * v_st.v / (v_params.omega0 * rGamma))
    f0 = v_st.v * v_params.c1 / rGamma + v_params.c0
    ta = f1 - f0
    ta = jnp.where((jnp.abs(v_st.v - 0) < 1e-9) & (ta < 0), 0, ta)

    v_update = v_st.v + ((rGamma / v_params.I) * ta) * v_params.step

    return VehStates(x_update, y_update, theta_update, v_update)

# Main function to run the simulation
def main():
    # Initialize driver data
    m_data = []
    driver_input(m_data, 'jax/inputs/acc3.txt')  # Replace with your file path

    # Vehicle parameters and initial state
    v_params = VehParam(step=0.001, l=2.5, R=0.3, gamma=1.0, tau0=100, omega0=100, c1=0.5, c0=0.1, I=1500, steerMap=1.0)
    v_st = VehStates(x=0, y=0, theta=0, v=0)
    controls = [0, 0, 0, 0]

    endTime = 10.0
    t = 0

    # Start timer
    start_time = time.time()

    # Simulation loop
    while t < endTime:
        get_controls(controls, m_data, t)
        v_st = integrate_any(v_st, v_params, controls[1:4])

        t += v_params.step

    # End timer
    end_time = time.time()

    # Calculate and print duration
    duration_sec = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Simulation Duration: {duration_sec} ms")

if __name__ == "__main__":
    main()

