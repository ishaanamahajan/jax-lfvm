import jax
import jax.numpy as jnp
from collections import namedtuple

# Define the vehicle's state and parameters
VehicleState = namedtuple('VehicleState', ['v', 'r', 'phi', 'theta'])
VehicleParam = namedtuple('VehicleParam', ['m', 'Iz', 'Iy', 'Ix', 'a', 'b', 'c', 'd'])

def compute_lateral_forces(v_state, v_param):
    
    Fyf = 0  # Front lateral force
    Fyr = 0  # Rear lateral force
    return Fyf, Fyr

def vehicle_dynamics(v_state, v_param, delta):
    """
    Compute the next state of the vehicle.
    
    :param v_state: VehicleState, current state of the vehicle
    :param v_param: VehicleParam, parameters of the vehicle
    :param delta: float, steering angle (rad)
    :return: VehicleState, the next state of the vehicle
    """

    # Compute lateral forces
    Fyf, Fyr = compute_lateral_forces(v_state, v_param)

    # Simplified vehicle dynamics equations
    v_dot = (Fyf + Fyr) / v_param.m
    r_dot = (v_param.a * Fyf - v_param.b * Fyr) / v_param.Iz
    phi_dot = v_state.r
    theta_dot = v_state.v / v_param.c  # Assuming c is the wheelbase

    # Update state
    new_v_state = VehicleState(
        v = v_state.v + v_dot,
        r = v_state.r + r_dot,
        phi = v_state.phi + phi_dot,
        theta = v_state.theta + theta_dot
    )

    return new_v_state


v_state = VehicleState(v=0, r=0, phi=0, theta=0)
v_param = VehicleParam(m=1500, Iz=1500, Iy=800, Ix=500, a=1.5, b=1.5, c=2.5, d=2.5)
delta = 0.1  # Steering angle in radians


new_v_state = vehicle_dynamics(v_state, v_param, delta)
