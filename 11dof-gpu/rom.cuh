
#ifndef ROM_CUH
#define ROM_CUH

// Header file for structs and functions for the reduced order model




// some constants used throughout
static const double C_PI = 3.141592653589793238462643383279;
static const double C_2PI = 6.283185307179586476925286766559;
static const double G = 9.81; // gravity constant
static const double rpm2rad = C_PI / 30; // converting RPM to radians

// error checking macros

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
};


// #define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
// void checkLast(const char* const file, const int line)
// {
//     cudaError_t err{cudaGetLastError()};
//     if (err != cudaSuccess)
//     {
//         std::cerr << "CUDA Runtime Error at: " << file << ":" << line
//                   << std::endl;
//         std::cerr << cudaGetErrorString(err) << std::endl;
//         // We don't exit when we encounter CUDA errors in this example.
//         // std::exit(EXIT_FAILURE);
//     }
// }




// Structure of entries as this is how we represent the driver data
struct Entry{
    __device__ __host__ Entry() {} // alias
    // constructor
    __device__ __host__ Entry(double time, double steering, double throttle, double braking)
        : m_time(time), m_steering(steering), m_throttle(throttle), m_braking(braking) {}

    double m_time;
    double m_steering;
    double m_throttle;
    double m_braking;
};

/// Structure for entries of any kind of map (anything with x and y)
struct MapEntry{
    __device__ __host__ MapEntry() {} // alias
    //constructor
    __device__ __host__ MapEntry(double x, double y)
        : _x(x), _y(y) {}
    

    double _x; // x coordinate of the map
    double _y; // y coordinate of the map
};


// Define our primary structures


// Reduced order vehicle model parameters
struct VehicleParam{
    
    // default constructor with pre tuned values from HMMVW calibration
    VehicleParam() 
        : _a(1.6889), _b(1.6889), _h(0.713), _m(2097.85), _jz(4519.), _muf(127.866),
        _mur(129.98), _nonLinearSteer(false), _maxSteer(0.6525249), _crankInertia(1.1), _upshift_RPS(10000),
        _downshift_RPS(0), _noGears(1), _tcbool(false), _maxBrakeTorque(4000.), 
        _step(1e-2), _steerMapSize(5), _lossMapSize(5), _cfMapSize(5), _trMapSize(5), _throttleMod(0)   {}
    
    
    // constructor
    VehicleParam(double a, double b, double h, double m, double Jz
        , double muf, double mur, double hrcf, double hrcr, bool steer_bool, double maxSteer, double crank_inertia, double up_RPS, double down_RPS, int no_gears, bool tc_bool,
        double maxTorque, double brakeTorque, double maxSpeed, double step, double steer_map_size, double tor_map_size, int loss_map_size,
        int cf_map_size, int tr_map_size, bool throttle_mod)
        : _a(a), _b(b), _h(h), _m(m), _jz(Jz), _muf(muf), _mur(mur), _nonLinearSteer(steer_bool), _maxSteer(maxSteer), _crankInertia(crank_inertia),
        _upshift_RPS(up_RPS), _downshift_RPS(down_RPS), _noGears(no_gears), _tcbool(tc_bool), _maxBrakeTorque(brakeTorque),
        _step(step), _steerMapSize(steer_map_size),_torMapSize(tor_map_size), _lossMapSize(loss_map_size), _cfMapSize(cf_map_size),
        _trMapSize(tr_map_size),  _throttleMod(throttle_mod) {}


    double _a, _b; // distance c.g. - front axle & distance c.g. - rear axle (m)
    double _h; // height of c.g
    double _m; // total vehicle mass (kg)
    double _jz; // yaw moment inertia (kg.m^2)
    double _muf,_mur; // front and rear unsprung mass

    bool _nonLinearSteer; 
    // Non linear steering map in case the steering mechanism is not linear
    MapEntry  *_steerMap;
    int _steerMapSize;

    // max steer angle parameters of the vehicle
    double _maxSteer;

    // crank shaft inertia
    double _crankInertia;
    // some gear parameters
    double _upshift_RPS;
    double _downshift_RPS;
    int _noGears; // Number of gears

    // boolean for torque converter presense
    bool _tcbool;

    bool _throttleMod; 


    
    double _maxBrakeTorque; // max brake torque
    
    double _step; // vehicle integration time step

    double *_gearRatios; // gear ratios
    // We will always define the powertrain with a map 
    MapEntry *_powertrainMap;
    int _torMapSize;
    MapEntry *_lossesMap;
    int _lossMapSize;

    // torque converter maps
    MapEntry *_CFmap; // capacity factor map
    int _cfMapSize;
    MapEntry *_TRmap; // Torque ratio map
    int _trMapSize;

    
};

// The reduced order vehicle states - only accessed from the device as we run the simulation there
struct VehicleState{
    
    // default constructor just assigns zero to all members
    __device__ __host__ VehicleState() 
        : _x(0.), _y(0.), _u(0.), _v(0.), _psi(0.), _wz(0.), _udot(0.), _vdot(0.), _wzdot(0.),
        _fzf(0.), _fzr(0.), _tor(0.), _crankOmega(0.), _current_gr(0), _tc_reverse_flow(false)  {}


    // special constructor in case need to start simulation
    // from some other state
    double _x, _y; // x and y position
    double _u, _v; // x and y velocity
    double _psi, _wz; // yaw angle and yaw rate
    

    // acceleration 'states'
    double _udot, _vdot; 
    double _wzdot;


    // vertical forces on each tire
    double _fzf, _fzr;

    // crank torque (used to transmit torque to tires) and crank angular velocity state
    double _tor;
    double _crankOmega;
    int _current_gr;
    double _tc_inp_tor;
    double _tc_out_tor;
    double _tc_out_omg;
    bool _tc_reverse_flow;
    double _sr;
};


// Tire parameters and states
struct TMeasyParam{

    // constructor that takes default values of HMMWV
    TMeasyParam()
        : _jw(6.69), _rr(0.015), _mu(0.8), _r0(0.4699), _pn(8562.8266),
        _pnmax(29969.893), _cx(185004.42), _cy(164448.37),  _kt(411121.0), 
        _dx(3700.), _dy(3488.), _rdyncoPn(0.375), _rdyncoP2n(0.75), _fzRdynco(0), _rdyncoCrit(0),
        _dfx0Pn(151447.29), _dfx0P2n(236412.79), _fxmPn(7575.3614), _fxmP2n(12808.276),
        _fxsPn(4657.9208), _fxsP2n(8625.3352), _sxmPn(0.12), _sxmP2n(0.15),
        _sxsPn(0.9), _sxsP2n(0.95), _dfy0Pn(50931.693), _dfy0P2n(94293.847),
        _fymPn(6615.0404), _fymP2n(12509.947), _fysPn(6091.5092), _fysP2n(11443.875),
        _symPn(0.38786), _symP2n(0.38786), _sysPn(0.82534), _sysP2n(0.91309), _step(1e-2) {}

    // constructor that takes given values - ugly looking code - can this be beutified?
    TMeasyParam(double jw, double rr, double mu, double r0, double pn,
        double pnmax, double cx, double cy, double dx, double dy, double kt, 
        double rdyncoPn, double rdyncoP2n, double fzRdynco, double rdyncoCrit, double dfx0Pn, double dfx0P2n, 
        double fxmPn, double fxmP2n, double fxsPn, double fxsP2n, double sxmPn, 
        double sxmP2n, double sxsPn, double sxsP2n, double dfy0Pn, double dfy0P2n, 
        double fymPn, double fymP2n, double fysPn, double fysP2n, double symPn, 
        double symP2n, double sysPn, double sysP2n, double step)
        : _jw(jw), _rr(rr), _mu(mu), _r0(r0), _pn(pn),
        _pnmax(pnmax), _cx(cx), _cy(cy), _kt(kt), _dx(dx),
        _dy(dy), _rdyncoPn(rdyncoPn), _rdyncoP2n(rdyncoP2n), _fzRdynco(fzRdynco),
        _rdyncoCrit(rdyncoCrit), _dfx0Pn(dfx0Pn), _dfx0P2n(dfx0P2n), _fxmPn(fxmPn), _fxmP2n(fxmP2n),
        _fxsPn(fxsPn), _fxsP2n(fxsP2n), _sxmPn(sxmPn), _sxmP2n(sxmP2n),
        _sxsPn(sxsPn), _sxsP2n(sxsP2n), _dfy0Pn(dfy0Pn), _dfy0P2n(dfy0P2n),
        _fymPn(fymPn), _fymP2n(fymP2n), _fysPn(fysPn), _fysP2n(fysP2n),
        _symPn(symPn), _symP2n(symP2n), _sysPn(sysPn), _sysP2n(sysP2n), _step(step) {}


    // basic tire parameters
    double _jw; // wheel inertia
    double _rr; // rolling resistance of tire
    double _mu; // friction constant
    double _r0; // unloaded tire radius

    // TM easy specific tire params
    double _pn, _pnmax; // nominal and max vertical force
    double _cx, _cy, _kt; // longitudinal, lateral and vertical stiffness
    double _dx, _dy; // longitudinal and lateral damping coeffs. No vertical damping


    // TM easy force characteristic params 
    // 2 values - one at nominal load and one at max load

    // dynamic radius weighting coefficient and a critical value for the vertical force
    double _rdyncoPn, _rdyncoP2n, _fzRdynco, _rdyncoCrit;

    // Longitudinal
    double _dfx0Pn, _dfx0P2n; // intial longitudinal slopes dFx/dsx [N]
    double _fxmPn,_fxmP2n; // maximum longituidnal force [N]
    double _fxsPn, _fxsP2n; // Longitudinal load at sliding [N]
    double _sxmPn, _sxmP2n; // slip sx at maximum longitudinal load Fx
    double _sxsPn, _sxsP2n; // slip sx where sliding begins

    // Lateral
    double _dfy0Pn, _dfy0P2n; // intial lateral slopes dFx/dsx [N]
    double _fymPn,_fymP2n; // maximum lateral force [N]
    double _fysPn, _fysP2n; // Lateral load at sliding [N]
    double _symPn, _symP2n; // slip sx at maximum lateral load Fx
    double _sysPn, _sysP2n; // slip sx where sliding begins

    double _step; // integration time step




};

struct TMeasyState{
    // default contructor to 0's
    __device__ __host__ TMeasyState ()
        : _xe(0.), _ye(0.), _xedot(0.), _yedot(0.), _omega(0.),
        _xt(0.), _rStat(0.), _fx(0.), _fy(0.), _fz(0.), _vsx(0.), _vsy(0.), _My(0.), _engTor(0.) {}


    // special constructor in case we want to start the simualtion at
    // some other time step
    __device__ __host__ TMeasyState(double xe, double ye, double xedot, double yedot,
        double omega, double xt, double rStat, double fx, double fy,
        double fz, double vsx, double vsy, double init_ratio)
        : _xe(xe), _ye(ye), _xedot(xedot), _yedot(yedot),
        _omega(omega), _xt(xt), _rStat(rStat), _fx(fx), _fy(fy),
        _fz(fz), _vsx(vsx), _vsy(vsy) {}


    // the actual state that are intgrated
    double _xe, _ye; // long and lat tire deflection
    double _xedot, _yedot; // long and lat tire deflection velocity
    double _omega; // angular velocity of wheel


    // other "states" that we need to keep track of
    double _xt; // vertical tire compression
    double _rStat; // loaded tire radius
    double _fx, _fy, _fz; // long, lateral and vertical force in tire frame

    // velocities in tire frame
    double _vsx, _vsy;

    double _My; // Rolling resistance moment (negetive)

    // torque from engine that we keep track of
    double _engTor;

};



// The entire pacakge of information needed for a vehicle simulation - All the parameters
struct SimData{
    
    SimData() {} //alias

    //constructor
    SimData(VehicleParam veh_param, TMeasyParam tire_param, Entry *driver_data)
        : _veh_param(veh_param), _tire_param(tire_param), _driver_data(driver_data) {}


    VehicleParam _veh_param; // we need the vehicle parameters
    TMeasyParam _tire_param; // we need the tire parameters
    Entry *_driver_data; // we need the vehicles controls which is an array of driverData entries
    unsigned int _driver_data_len; // I need this in the getControls function.. or maybe I don't... but now I do
};

// Structure for all the state information needed. This is passed around each kernel call
struct SimState{
    
    SimState() {} //alias

    //constructor
    SimState(VehicleState veh_st, TMeasyState tiref_state, TMeasyState tirer_state )
        : _veh_st(veh_st), _tiref_state(tiref_state), _tirer_state(tirer_state) {}


    // vehicle state
    VehicleState _veh_st; 
    TMeasyState _tiref_state; 
    TMeasyState _tirer_state; 
    
};


// Some utils for interpolation and the like
__device__ __host__ inline double InterpL(double fz, double w1, double w2, double pn) { return w1 + (w2 - w1) * (fz / pn - 1.); }
__device__ __host__ inline double InterpQ(double fz, double w1, double w2, double pn) { return (fz/pn) * (2. * w1 - 0.5 * w2 - (w1 - 0.5 * w2) * (fz/pn)); }
__device__ double sineStep(double x, double x1, double y1, double x2, double y2);
template <typename T> __device__ int sgn(T val) { return (T(0) < val) - (val < T(0)); }

// clamp function from chrono
template <typename T>
__device__ T clamp(T value, T limitMin, T limitMax) {
    if (value < limitMin)
        return limitMin;
    if (value > limitMax)
        return limitMax;

    return value;
};




// Function to read input control file into a driver data structure
// here we continue using some c++ stuff as this only runs on the host
// for now and its convinient. Users duty to modify this vector of entries
// into a pointer 
__host__ void driverInput(std::vector <Entry>& m_data ,const std::string& filename);



// Functions to set these tire and vehicle parameters from a JSON file
// Only runs on host for now
__host__ void setVehParamsJSON(VehicleParam& v_params, const char *fileName);
__host__ void setTireParamsJSON(TMeasyParam& t_params, const char *fileName);


// kernel function
__global__ void run(const SimData* simData, double* response, double current_time, const double endTime, const double step, const unsigned int collection_rate ,int states, 
                        unsigned int num_vehicles, SimState *sim_state);

// custom lower bound function for entry and map entry
__device__ unsigned int lower_bound(Entry *a, unsigned int n, Entry x);
__device__ unsigned int lower_bound_map(MapEntry *a, unsigned int n, MapEntry x);

// device function to get the controls 
__device__ void getControls(double *controls, Entry *m_data, const double time,const unsigned int len);


// device function to get the Y coordiante of the map 
__device__ double getMapY(const MapEntry *map, const double motor_speed, const unsigned int size);

// Tire and vehicle initialization functions
__device__ __host__ void vehInit(VehicleState  *v_state, const VehicleParam *v_params);
__device__ __host__ void tireInit(TMeasyParam *t_params);


// Frame transform functions
__device__ void vehToTireTransform(TMeasyState *tiref_st,TMeasyState *tirer_st,
                            const VehicleState *v_states, const VehicleParam *v_params, const double *controls);

__device__ void tireToVehTransform(TMeasyState *tiref_st,TMeasyState *tirer_st,
                            const VehicleState *v_states, const VehicleParam *v_params, const double *controls);

// Function to provide a differential split for left and right wheels (for both front and rear)
 __device__ void differentialSplit(double torque,
                       double max_bias,
                       double speed_left,
                       double speed_right,
                       double *torque_left,
                       double *torque_right,
                       bool split);

// Function that evaluates the powertrain
__device__ void evalPowertrain(VehicleState *v_states, TMeasyState *tiref_st,
                    TMeasyState *tirer_st, VehicleParam *v_params, const TMeasyParam *t_params,
                    const double *controls);
                    


// curve fitting for slip vs force used in tireAdv
__device__ void tmxy_combined(double *f, double *fos, double s, double df0, double sm, double fm, double ss, double fs);

// drive and brake torque functions
__device__ double driveTorque(const VehicleParam *v_params, const double throttle, const double omega);

__device__ inline double brakeTorque(const VehicleParam *v_params, const double brake){
    return v_params->_maxBrakeTorque * brake;
}




// Tire advance function
__device__ void tireAdv(TMeasyState *t_states, const TMeasyParam *t_params, const VehicleState *v_states, 
                const VehicleParam *v_params, const double *controls);


// vehicle advance function
// function to advance the time step of the vehicle
__device__ void vehAdv(VehicleState *v_states, const VehicleParam *v_params, 
            const double *fx, const double *fy, const double huf, const double hur);


//CSV dump functions
__host__ void dump_csv_all(unsigned int num, unsigned int states, double *host_response, unsigned int host_collection_steps, 
                            unsigned int dump_no);
__host__ void dump_csv_which(unsigned int num_vehicles, unsigned int states, double *host_response, unsigned int host_collection_steps, 
                            unsigned int dump_no, int *which_outs,unsigned int no_outs);

#endif