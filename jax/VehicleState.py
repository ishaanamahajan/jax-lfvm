class VehicleState:
    def __init__(self, x=0.0, y=0.0, dx=0.0, dy=0.0, u=0.0, v=0.0, psi=0.0, wz=0.0, phi=0.0, wx=0.0, udot=0.0, vdot=0.0, wxdot=0.0, wzdot=0.0, tor=0.0, crankOmega=0.0, current_gr=0):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.u = u
        self.v = v
        self.psi = psi
        self.wz = wz
        self.phi = phi
        self.wx = wx
        self.udot = udot
        self.vdot = vdot
        self.wxdot = wxdot
        self.wzdot = wzdot
        self.tor = tor
        self.crankOmega = crankOmega
        self.current_gr = current_gr


    def to_array(self):
        return jnp.array([self.x, self.y, self.dx, self.dy, self.u, self.v, self.psi, self.wz, self.phi, self.wx, self.udot, self.vdot, self.wxdot, self.wzdot, self.tor, self.crankOmega, self.current_gr])
