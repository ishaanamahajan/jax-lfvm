import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from vehicle_state import VehicleState
from vehicle_param import VehicleParam

def computeVehRHS(v_states, v_params, fx, fy):
    # Constants
    G = 9.81  # Assuming G is the gravitational constant

    # get the total mass of the vehicle and the vertical distance from the sprung mass C.M. to the vehicle
    mt = v_params.m + 2 * (v_params.muf + v_params.mur)
    hrc = (v_params.hrcf * v_params.b + v_params.hrcr * v_params.a) / (v_params.a + v_params.b)

    # a bunch of variables to simplify the formula
    E1 = -mt * v_states.wz * v_states.u + jnp.sum(fy)

    E2 = (fy[0] + fy[1]) * v_params.a - (fy[2] + fy[3]) * v_params.b + (fx[1] - fx[0]) * v_params.cf / 2 + \
         (fx[3] - fx[2]) * v_params.cr / 2 + \
         (-v_params.muf * v_params.a + v_params.mur * v_params.b) * v_states.wz * v_states.u

    E3 = v_params.m * G * hrc * v_states.phi - (v_params.krof + v_params.kror) * v_states.phi - \
         (v_params.brof + v_params.bror) * v_states.wx + hrc * v_params.m * v_states.wz * v_states.u

    A1 = v_params.mur * v_params.b - v_params.muf * v_params.a
    A2 = v_params.jx + v_params.m * hrc ** 2
    A3 = hrc * v_params.m

    # update the acceleration states - level 2 variables
    v_states.udot = v_states.wz * v_states.v + \
                    (1 / mt) * (jnp.sum(fx) + (-v_params.mur * v_params.b + v_params.muf * v_params.a) * v_states.wz ** 2 - \
                    2. * hrc * v_params.m * v_states.wz * v_states.wx)

    # common denominator
    denom = A2 * A1 ** 2 - 2. * A1 * A3 * v_params.jxz + v_params.jz * A3 ** 2 + \
            mt * v_params.jxz ** 2 - A2 * v_params.jz * mt

    v_states.vdot = (E1 * v_params.jxz ** 2 - A1 * A2 * E2 + A1 * E3 * v_params.jxz + \
                     A3 * E2 * v_params.jxz - A2 * E1 * v_params.jz - A3 * E3 * v_params.jz) / denom

    v_states.wxdot = (A1 ** 2 * E3 - A1 * A3 * E2 + A1 * E1 * v_params.jxz - A3 * E1 * v_params.jz + \
                      E2 * v_params.jxz * mt - E3 * v_params.jz * mt) / denom

    v_states.wzdot = (A3 ** 2 * E2 - A1 * A2 * E1 - A1 * A3 * E3 + A3 * E1 * v_params.jxz - A2 * E2 * mt + \
                      E3 * v_params.jxz * mt) / denom

    # update the level 0 variables using the next time step level 1 variables
    v_states.dx = v_states.u * jnp.cos(v_states.psi) - v_states.v * jnp.sin(v_states.psi)
    v_states.dy = v_states.u * jnp.sin(v_states.psi) + v_states.v * jnp.cos(v_states.psi)

    return v_states

def grad_computeVehRHS(v_states, v_params, fx, fy):

     fx = jnp.array(fx)

 
    return grad(computeVehRHS, argnums=(0,1,2,3))(v_states, v_params, fx, fy)


