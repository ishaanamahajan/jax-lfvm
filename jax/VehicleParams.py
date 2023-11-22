class VehicleParam:
    def __init__(self, a=1.6889, b=1.6889, h=0.713, m=2097.85, jz=4519., jx=1289., jxz=3.265, cf=1.82, cr=1.82, muf=127.866, mur=129.98, hrcf=0.379, hrcr=0.327, krof=31000, kror=31000, brof=3300, bror=3300, nonLinearSteer=False, maxSteer=0.6525249, crankInertia=1.1, tcbool=False, maxBrakeTorque=4000., c1=0., c0=0., step=1e-2, throttleMod=0, driveType=1, whichWheels=1):
        self.a = a
        self.b = b
        self.h = h
        self.m = m
        self.jz = jz
        self.jx = jx
        self.jxz = jxz
        self.cf = cf
        self.cr = cr
        self.muf = muf
        self.mur = mur
        self.hrcf = hrcf
        self.hrcr = hrcr
        self.krof = krof
        self.kror = kror
        self.brof = brof
        self.bror = bror
        self.nonLinearSteer = nonLinearSteer
        self.maxSteer = maxSteer
        self.crankInertia = crankInertia
        self.tcbool = tcbool
        self.maxBrakeTorque = maxBrakeTorque
        self.c1 = c1
        self.c0 = c0
        self.step = step
        self.throttleMod = throttleMod
        self.driveType = driveType
        self.whichWheels = whichWheels


    
    
       
    def to_array(self):
        return jnp.array([self.a, self.b, self.h, self.m, self.jz, self.jx, self.jxz, self.cf, self.cr, self.muf, self.mur, self.hrcf, self.hrcr, self.krof, self.kror, self.brof, self.bror, float(self.nonLinearSteer), self.maxSteer, self.crankInertia, float(self.tcbool), self.maxBrakeTorque, self.c1, self.c0, self.step, self.throttleMod, self.driveType, self.whichWheels])
