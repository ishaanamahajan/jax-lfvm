#include <cuda.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "third_party/rapidjson/document.h"
#include "third_party/rapidjson/filereadstream.h"
#include "rom.cuh"
#include "csvwriter.h"




// The simulation kernel
// Takes the simData as input and returns a double pointer.
// response is a double device pointer and this is how it is structured:
// The response of the vehicle is stored in row major order i.e.
// Timestep 1 [x,y,u,v,phi,psi,wx,wz,lf_omega,rf_omega,lr_omega,rr_omega] , Timestep 2 [x,y,u,v,phi,psi,wx,wz,lf_omega,rf_omega,lr_omega,rr_omega]
// and so on of the first vehicle... then so on for all the other vehicles
// So the entire size to be allocated for responses is num_vehicles * num_states * num_timesteps
// need to do profiling to undertand the best way to store this
__global__ void run(const SimData* simData, double* response, double current_time, const double endTime, const double step, const unsigned int collection_rate ,int states, 
                        unsigned int num_vehicles, SimState *simState){


    // the simulation number of the vehilce (overall thread number)
    unsigned int sim_no =  blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread extracts its simulation Data
    VehicleParam veh_param = simData[sim_no]._veh_param;
    TMeasyParam tire_param = simData[sim_no]._tire_param;
    Entry *driver_data = simData[sim_no]._driver_data;
    unsigned int len = simData[sim_no]._driver_data_len;

    // Each thread also has its own sim state
    VehicleState veh_st = simState[sim_no]._veh_st;
    TMeasyState tiref_st = simState[sim_no]._tiref_state;
    TMeasyState tirer_st = simState[sim_no]._tirer_state;
    

    unsigned int timeStepNo = 0; // Keep track of our timesteps for storing output
    unsigned int collectionStepNo = 0;


    // initialize front_controls and rear_controls
    double front_controls[4];
    double t = current_time;
    // Run the simulation loop
    while(t < (endTime - step/10.)){

        
        // get front controls
        getControls(&front_controls[0], driver_data, t, len);
        // rear controls just does not have steering - everything else is the same
        double rear_controls[4] = {front_controls[0], 0, front_controls[2], front_controls[3]};

        //transform vehicle quantities to tire as we advance the tire first
        vehToTireTransform(&tiref_st,&tirer_st, &veh_st, &veh_param, &front_controls[0]);

        // advnace our 4 tires
        tireAdv(&tiref_st, &tire_param, &veh_st, &veh_param, &front_controls[0]);
        tireAdv(&tirer_st, &tire_param, &veh_st, &veh_param, &rear_controls[0]);
        

        // Evalaute the powertrain and advance the tire angular velocities and the angular
        // velocity of the crank shaft (if we have Torque converter on)
      
        evalPowertrain(&veh_st, &tiref_st, &tirer_st, &veh_param, &tire_param, &front_controls[0]);
        //transform tire forces to vehicle forces
        tireToVehTransform(&tiref_st, &tirer_st, &veh_st, &veh_param, &front_controls[0]);

        // take out the forces
        double fx[2] = {tiref_st._fx,tirer_st._fx};
        double fy[2] = {tiref_st._fy,tirer_st._fy};

        // 2 other weird parameters
        double huf = tiref_st._rStat;
        double hur = tirer_st._rStat;

        // time to advance the vehicle
        vehAdv(&veh_st, &veh_param, &fx[0], &fy[0], huf, hur);


        // imcrement time by step
        t += step;
        timeStepNo++;
        // collect every collection rate steps
        if(timeStepNo % collection_rate == 0){
            unsigned int time_offset = collectionStepNo * (states * num_vehicles);

            response[time_offset + (num_vehicles * 0) + sim_no] = t;
            // printf("%f, %d \n", t, time_offset + (num_vehicles * 0) + sim_no);
            response[time_offset + (num_vehicles * 1) + sim_no] = veh_st._x;
            response[time_offset + (num_vehicles * 2) + sim_no] = veh_st._y;
            response[time_offset + (num_vehicles * 3) + sim_no] = veh_st._u;
            response[time_offset + (num_vehicles * 4) + sim_no] = veh_st._v;
            response[time_offset + (num_vehicles * 5) + sim_no] = veh_st._psi;
            response[time_offset + (num_vehicles * 6) + sim_no] = veh_st._wz;
            response[time_offset + (num_vehicles * 7) + sim_no] = tiref_st._omega;
            response[time_offset + (num_vehicles * 8) + sim_no] = tirer_st._omega;
            

            collectionStepNo++;
        }
    }
    // move the memory back into simState for the next time step
    simState[sim_no]._veh_st = veh_st;
    simState[sim_no]._tiref_state = tiref_st;
    simState[sim_no]._tirer_state = tirer_st;
    
}


// Setting vehicle and tire parameters from JSON files

__host__ void setVehParamsJSON(VehicleParam& v_params, const char *fileName){

    // Open the file
    FILE* fp = fopen(fileName,"r");

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    // parse the stream into DOM tree
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);


    if (d.HasParseError()) {
        std::cout << "Error with rapidjson:" << std::endl << d.GetParseError() << std::endl;
    }

    // the file should have all these parameters defined
    v_params._a = d["a"].GetDouble();
    v_params._b = d["b"].GetDouble();
    v_params._m = d["m"].GetDouble();
    v_params._h = d["h"].GetDouble();
    v_params._jz = d["jz"].GetDouble();
    
    v_params._muf = d["muf"].GetDouble();
    v_params._mur = d["mur"].GetDouble();
    v_params._maxBrakeTorque = d["maxBrakeTorque"].GetDouble();
    v_params._step = d["step"].GetDouble();
    v_params._tcbool = d["tcBool"].GetBool(); // check if we have a torque converter
    v_params._upshift_RPS = d["upshiftRPM"].GetDouble() * rpm2rad;
    v_params._downshift_RPS = d["downshiftRPM"].GetDouble() * rpm2rad;
    v_params._throttleMod = d["throttleMod"].GetBool();


      // Non linear steering which maps the normalized steering input to wheel angle
    v_params._nonLinearSteer = d["nonLinearSteer"].GetBool();
    if(v_params._nonLinearSteer){
        int steer_map_size = d["steerMap"].Size();
        CHECK_CUDA_ERROR(cudaMallocManaged((void **)&v_params._steerMap, sizeof(double) * steer_map_size));
        for(int i = 0; i < steer_map_size; i++){
            MapEntry m;
            m._x = d["steerMap"][i][0u].GetDouble();
            m._y = d["steerMap"][i][1u].GetDouble();
            v_params._steerMap[i] = m;
        }
        v_params._steerMapSize = steer_map_size;
    }
    // If there is no non linear steer then the normalized steering input is just multiplied by the max steering wheel angle
    else{
        
        v_params._maxSteer = d["maxSteer"].GetDouble();
    }



    

    // read the gear ratios
    int gears = d["gearRatios"].Size();
    CHECK_CUDA_ERROR(cudaMallocManaged((void **)&v_params._gearRatios, sizeof(double) * gears)); // assign the memory for the gears
    for(int i = 0; i < gears; i++){
        v_params._gearRatios[i] = d["gearRatios"][i].GetDouble();
    }
    v_params._noGears = gears;




    // assign memory for the torque maps and fill up all the values
    int tor_map_size = d["torqueMap"].Size();
    CHECK_CUDA_ERROR(cudaMallocManaged((void **)&v_params._powertrainMap, sizeof(MapEntry) * tor_map_size)); // assign the memory for the torque map
    for(int i = 0; i < tor_map_size; i++){
        MapEntry m;
        m._x = d["torqueMap"][i][0u].GetDouble()*rpm2rad;
        m._y = d["torqueMap"][i][1u].GetDouble();
        v_params._powertrainMap[i] = m;
    }
    v_params._torMapSize = tor_map_size;

    // assign memory for the losses maps and fill up all the values
    int loss_map_size = d["lossesMap"].Size();
    CHECK_CUDA_ERROR(cudaMallocManaged((void **)&v_params._lossesMap, sizeof(MapEntry) * loss_map_size)); // assign the memory for the torque map
    for(int i = 0; i < loss_map_size; i++){
        MapEntry m;
        m._x = d["lossesMap"][i][0u].GetDouble()*rpm2rad;
        m._y = d["lossesMap"][i][1u].GetDouble();
        v_params._lossesMap[i] = m;
    }
    v_params._lossMapSize = loss_map_size;




    // Now assign memory for the torque converter maps if we have them 
    if(v_params._tcbool){
        v_params._crankInertia = d["crankInertia"].GetDouble();
        int cf_map_size = d["capacityFactorMap"].Size();
        CHECK_CUDA_ERROR(cudaMallocManaged((void **)&v_params._CFmap, sizeof(MapEntry) * cf_map_size));
        for(int i = 0; i < cf_map_size; i++){
            MapEntry m;
            m._x = d["capacityFactorMap"][i][0u].GetDouble();
            m._y = d["capacityFactorMap"][i][1u].GetDouble();
            v_params._CFmap[i] = m;
        }
        v_params._cfMapSize = cf_map_size;

        int tr_map_size = d["torqueRatioMap"].Size();
        CHECK_CUDA_ERROR(cudaMallocManaged((void **)&v_params._TRmap, sizeof(MapEntry) * tr_map_size));
        for(int i = 0; i < tr_map_size; i++){
            MapEntry m;
            m._x = d["torqueRatioMap"][i][0u].GetDouble();
            m._y = d["torqueRatioMap"][i][1u].GetDouble();
            v_params._TRmap[i] = m;
        }
        v_params._trMapSize = tr_map_size;
    }



}

__host__ void setTireParamsJSON(TMeasyParam& t_params, const char *fileName){
    // Open the file
    FILE* fp = fopen(fileName,"r");

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    // parse the stream into DOM tree
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);


    if (d.HasParseError()) {
        std::cout << "Error with rapidjson:" << std::endl << d.GetParseError() << std::endl;
    }

    // We mutiply most of the parameters by 2 because we use 2 tires instead of 4
    t_params._jw = d["jw"].GetDouble() * 2.;
    t_params._rr = d["rr"].GetDouble() * 2.;
    t_params._r0 = d["r0"].GetDouble();
    t_params._pn = d["pn"].GetDouble() * 2.;
    t_params._pnmax = d["pnmax"].GetDouble() * 2.;
    t_params._cx = d["cx"].GetDouble() * 2.;
    t_params._cy = d["cy"].GetDouble() * 2.;
    t_params._kt = d["kt"].GetDouble() * 2.;
    t_params._dx = d["dx"].GetDouble() * 2.;
    t_params._dy = d["dy"].GetDouble() * 2.;
    t_params._rdyncoPn = d["rdyncoPn"].GetDouble();
    t_params._rdyncoP2n = d["rdyncoP2n"].GetDouble();
    t_params._fzRdynco = d["fzRdynco"].GetDouble() * 2.;
    t_params._rdyncoCrit = d["rdyncoCrit"].GetDouble() * 2.;

    t_params._dfx0Pn = d["dfx0Pn"].GetDouble() * 2.;
    t_params._dfx0P2n = d["dfx0P2n"].GetDouble() * 2.;
    t_params._fxmPn = d["fxmPn"].GetDouble() * 2.;
    t_params._fxmP2n = d["fxmP2n"].GetDouble() * 2.;
    t_params._fxsPn = d["fxsPn"].GetDouble() * 2.;
    t_params._fxsP2n = d["fxsP2n"].GetDouble() * 2.;
    t_params._sxmPn = d["sxmPn"].GetDouble();
    t_params._sxmP2n = d["sxmP2n"].GetDouble();
    t_params._sxsPn = d["sxsPn"].GetDouble();
    t_params._sxsP2n = d["sxsP2n"].GetDouble();

    t_params._dfy0Pn = d["dfy0Pn"].GetDouble() * 2.;
    t_params._dfy0P2n = d["dfy0P2n"].GetDouble() * 2.;
    t_params._fymPn = d["fymPn"].GetDouble() * 2.;
    t_params._fymP2n = d["fymP2n"].GetDouble() * 2.;
    t_params._fysPn = d["fysPn"].GetDouble() * 2.;
    t_params._fysP2n = d["fysP2n"].GetDouble() * 2.;
    t_params._symPn = d["symPn"].GetDouble();
    t_params._symP2n = d["symP2n"].GetDouble();
    t_params._sysPn = d["sysPn"].GetDouble();
    t_params._sysP2n = d["sysP2n"].GetDouble();

    t_params._step = d["step"].GetDouble();

}



// some utils that are not defined inline
__device__ double sineStep(double x, double x1, double y1, double x2, double y2){
    if (x <= x1)
        return y1;
    if (x >= x2)
        return y2;
    
    double dx = x2 - x1;
    double dy = y2 - y1;
    double y = y1 + dy * (x - x1) / dx - (dy / C_2PI) * sin(C_2PI * (x - x1) / dx);
    return y;
}




// function to read the input file and fill it into a vector of entries
// these entries will then be used to get the controls at each time step.
// For now this is a very CPU function with c++ but I think we can make it 
// into pure C if need be
__host__ void driverInput(std::vector <Entry>& m_data ,const std::string& filename){
    
    std::ifstream ifile(filename.c_str());
    std::string line;

    
    // get each line
    while(std::getline(ifile,line)){
        std::istringstream iss(line);
        
        double time, steering, throttle, braking;

        // put the stream into our varaibles
        iss >> time >> steering >> throttle >> braking;

        if (iss.fail())
            break;

        // push into our structure
        m_data.push_back(Entry(time,steering,throttle,braking));
    }

    ifile.close();
}

// Implementing lower bound c++ function in C with our custom container "Entry"
// Just a binary search
__device__ unsigned int lower_bound(Entry *a, unsigned int n, Entry x) {
    int l = 0;
    int h = n; // not sure if this should be n or n-1
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (x.m_time <= a[mid].m_time) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

__device__ unsigned int lower_bound_map(const MapEntry *a, unsigned int n, MapEntry x) {
    int l = 0;
    int h = n; // not sure if this should be n or n-1
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (x._x <= a[mid]._x) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

// conversion of the getControls function from c++ to c to be used from the kernel
__device__ void getControls(double *controls, Entry *m_data, const double time, const unsigned int len){
    
    
    
    // cases for when we are outside the time range (take the first value for before and last value for after)
    if(time <= m_data[0].m_time){
        controls[0] = m_data[0].m_time;
        controls[1] = m_data[0].m_steering;
        controls[2] = m_data[0].m_throttle;
        controls[3] = m_data[0].m_braking;
        return;
    } else if(time >= m_data[len-1].m_time){
        controls[0] = m_data[len-1].m_time;
        controls[1] = m_data[len-1].m_steering;
        controls[2] = m_data[len-1].m_throttle;
        controls[3] = m_data[len-1].m_braking;
        return;
    }


    // Now if we are within 2 entries, we need to interpolate to find the controls between
    // these entries

    // call the c implementation of finding the correct index
    int right = lower_bound(m_data,len,Entry(time,0,0,0));

    // the left index will just be the previous one
    int left = right - 1;

    // linear interpolations
    double tbar = (time - m_data[left].m_time) / (m_data[right].m_time - m_data[left].m_time);

    controls[0] = time;
    controls[1] = m_data[left].m_steering + tbar * (m_data[right].m_steering - m_data[left].m_steering);
    controls[2] = m_data[left].m_throttle + tbar * (m_data[right].m_throttle - m_data[left].m_throttle);
    controls[3] = m_data[left].m_braking + tbar * (m_data[right].m_braking - m_data[left].m_braking);

}

__device__ double getMapY(const MapEntry *map, const double motor_speed, const unsigned int size){
    // if speed is less than or more than the min max, return the min max
    if(motor_speed <= map[0]._x){
        return map[0]._y;
    } else if(motor_speed >= map[size - 1]._x){
        return map[size - 1]._y;
    }

    // get the required iterators
    int right = lower_bound_map(map,size,MapEntry(motor_speed,0));

    // left index is just the previous one
    int left = right - 1;

    // linear interplolation
    double mbar = (motor_speed - map[left]._x) / (map[right]._x - map[left]._x);

    return (map[left]._y + mbar * (map[right]._y - map[left]._y));

}


__device__ __host__ void vehInit(VehicleState  *v_state, const VehicleParam *v_params){
    
     v_state->_fzf = ((v_params->_m * G * v_params->_b) / (2 * (v_params->_a + v_params->_b))  + v_params->_muf * G);
    v_state->_fzr = ((v_params->_m * G * v_params->_a) / (2 * (v_params->_a + v_params->_b))  + v_params->_mur * G); 

}

__device__ __host__ void tireInit(TMeasyParam *t_params){
    
    // calculates some critical values that are needed
    t_params->_fzRdynco = (t_params->_pn * (t_params->_rdyncoP2n - 2.0 * t_params->_rdyncoPn + 1.)) /
                            (2. * (t_params->_rdyncoP2n - t_params->_rdyncoPn));

    t_params->_rdyncoCrit = InterpL(t_params->_fzRdynco, t_params->_rdyncoPn, t_params->_rdyncoP2n,t_params->_pn);

}


// Frame transform functions
__device__ void vehToTireTransform(TMeasyState *tiref_st,TMeasyState *tirer_st,
                            const VehicleState *v_states,  const VehicleParam *v_params, const double *controls){

                             // get the controls and time out
                            //double delta = controls[1] * v_params->_maxSteer;

                            double delta = 0;
                            if(v_params->_nonLinearSteer){
                                // Extract steer map
                                
                                MapEntry *steer_map = v_params->_steerMap;
                                delta = getMapY(steer_map,controls[1], v_params->_steerMapSize);
                            }
                            else{
                                delta = controls[1] * v_params->_maxSteer;

                            }
                            double throttle = controls[2];
                            double brake = controls[3];

                            tiref_st->_fz = v_states->_fzf;
                            tiref_st->_vsy = v_states->_v + v_states->_wz * v_params->_a;
                            tiref_st->_vsx = v_states->_u * cos(delta) + tiref_st->_vsy * sin(delta);

                            // Rear
                            tirer_st->_fz = v_states->_fzr;
                            tirer_st->_vsy = v_states->_v - v_states->_wz * v_params->_b;
                            tirer_st->_vsx = v_states->_u; // no steer


                                
                            }


__device__ void tireToVehTransform(TMeasyState *tiref_st,TMeasyState *tirer_st,
                            const VehicleState *v_states, const VehicleParam *v_params, const double *controls){


                             // get the controls and time out
                            //double delta = controls[1] * v_params->_maxSteer;
                            double delta = 0;
                            if(v_params->_nonLinearSteer){
                                // Extract steer map
                                
                                MapEntry *steer_map = v_params->_steerMap;
                                delta = getMapY(steer_map,controls[1], v_params->_steerMapSize);
                            }
                            else{
                                delta = controls[1] * v_params->_maxSteer;

                            }
                            double throttle = controls[2];
                            double brake = controls[3];                          
                                                  
                            
                            double _fx,_fy;

                            // left front
                            _fx = tiref_st->_fx * cos(delta) - tiref_st->_fy * sin(delta);
                            _fy = tiref_st->_fx * sin(delta) + tiref_st->_fy * cos(delta);
                            tiref_st->_fx = _fx;
                            tiref_st->_fy = _fy;

                                                          



                            }


__device__ double driveTorque(const VehicleParam *v_params, const double throttle, const double motor_speed){




    double motor_torque = 0.;
    // If we have throttle modulation like in a motor
    if(v_params->_throttleMod){
        MapEntry *power_train_map = v_params->_powertrainMap;
        int len = v_params->_torMapSize;
        for(int i = 0; i<len; i++){
            power_train_map[i]._x = v_params->_powertrainMap[i]._x * throttle;
            power_train_map[i]._y = v_params->_powertrainMap[i]._y * throttle;
        }
        // interpolate in the torque map to get the torque at this paticular speed
        motor_torque = getMapY(power_train_map, motor_speed,  v_params->_torMapSize);
        double motor_losses = getMapY(v_params->_lossesMap, motor_speed,  v_params->_lossMapSize);
        motor_torque = motor_torque + motor_losses;
    }
    else{ // Else we don't multiply the map but just the output torque
        motor_torque = getMapY(v_params->_powertrainMap, motor_speed, v_params->_torMapSize);
        double motor_losses = getMapY(v_params->_lossesMap, motor_speed,  v_params->_lossMapSize);
        motor_torque = motor_torque * throttle + motor_losses;

    }
    return motor_torque;

}

 __device__ void differentialSplit(double torque,
                       double max_bias,
                       double speed_rear,
                       double speed_front,
                       double *torque_rear,
                       double *torque_front, bool split) {

        double diff = abs(speed_rear - speed_front);

        if(split){
            // The bias grows from 1 at diff=0.25 to max_bias at diff=0.5
            double bias = 1;
            if (diff > 0.5)
                bias = max_bias;
            else if (diff > 0.25)
                bias = 4 * (max_bias - 1) * diff + (2 - max_bias);

            // Split torque to the slow and fast wheels.
            double alpha = bias / (1 + bias);
            double slow = alpha * torque;
            double fast = torque - slow;

            if (abs(speed_rear) < abs(speed_front)) {
                *torque_rear = slow;
                *torque_front = fast;
            } else {
                *torque_rear = fast;
                *torque_front = slow;
            }
        }
        else{
            *torque_rear = torque;
            *torque_front = 0;
        }
    
}


__device__ void evalPowertrain(VehicleState *v_states, TMeasyState *tiref_st,
                    TMeasyState *tirer_st, VehicleParam *v_params, const TMeasyParam *t_params,
                    const double *controls){

                        // get controls
                        double throttle = controls[2];
                        double brake = controls[3];


                        // some variables needed outside
                        double torque_t = 0;
                        double max_bias = 2;
                        // If we have a torque converter
                        if(v_params->_tcbool){
                            // set reverse flow to false at each timestep
                            v_states->_tc_reverse_flow = false;
                            // Split the angular velocities all the way uptill the gear box. All from previous time step
                            double omega_t = 0.5 * (tiref_st->_omega + tirer_st->_omega);
                            
                            // get the angular velocity at the torque converter wheel side 
                            // Note, the gear includes the differential gear as well
                            double omega_out = omega_t / (v_params->_gearRatios[v_states->_current_gr]);

                            // Get the omega input to the torque from the engine from the previous time step
                            double omega_in = v_states->_crankOmega;


                            //extract maps

                            double sr, cf, tr;
                            if((omega_out < 1e-9) || (omega_in < 1e-9)){ // if we are at the start things can get unstable
                                sr = 0;
                                // Get capacity factor from capacity lookup table
                                cf = getMapY(v_params->_CFmap, sr, v_params->_cfMapSize);

                                // Get torque ratio from Torque ratio lookup table 
                                tr = getMapY(v_params->_TRmap, sr, v_params->_trMapSize);
                            }
                            else{
                                // speed ratio for torque converter
                                sr =  omega_out / omega_in;

                                // Check reverse flow
                                if(sr > 1.){
                                    sr = 1. - (sr - 1.);
                                    v_states->_tc_reverse_flow = true;
                                }

                                if(sr < 0){
                                    sr = 0;
                                }

                                // get capacity factor from lookup table
                                cf = getMapY(v_params->_CFmap, sr, v_params->_cfMapSize);

                                // Get torque ratio from Torque ratio lookup table 
                                tr = getMapY(v_params->_TRmap, sr, v_params->_trMapSize);                              
                            }
                            // torque applied to the crank shaft
                            double torque_in = -pow((omega_in / cf),2);

                            // if its reverse flow, this should act as a brake
                            if(v_states->_tc_reverse_flow){
                                torque_in = -torque_in;
                            }

                            // torque applied to the shaft from torque converter on the wheel side
                            double torque_out;
                            if(v_states->_tc_reverse_flow){
                                torque_out = -torque_in; 
                            }
                            else{
                                torque_out = -tr * torque_in ;
                            }

                            // Now torque after the transimission
                            torque_t = torque_out / v_params->_gearRatios[v_states->_current_gr];
                            if(std::abs((v_states->_u - 0) < 1e-9) && (torque_t < 0)){
                                torque_t = 0;
                            }
                            

                            /////// DEBUG
                            v_states->_tc_inp_tor = -torque_in;
                            v_states->_tc_out_tor = torque_out;
                            v_states->_tc_out_omg = omega_out;
                            v_states->_sr = sr;

                            //////// Integrate Crank shaft

                            double dOmega_crank = (1./v_params->_crankInertia) * (driveTorque(v_params, throttle, v_states->_crankOmega) + torque_in);

                            v_states->_crankOmega = v_states->_crankOmega + v_params->_step * dOmega_crank;


                            ////// Gear shift for the next time step -> Here we have to check the RPM of the shaft from the T.C
                            if(omega_out > v_params->_upshift_RPS){
                                
                                // check if we have enough gears to upshift
                                if(v_states->_current_gr < v_params->_noGears - 1){
                                    v_states->_current_gr++;
                                }
                            }
                            // downshift
                            else if(omega_out < v_params->_downshift_RPS){
                                // check if we can down shift
                                if(v_states->_current_gr > 0){
                                    v_states->_current_gr--;
                                }
                            }                           


                        }
                        else{ // if there is no torque converter, things are simple

                            v_states->_crankOmega = 0.5 * (tiref_st->_omega + tirer_st->_omega)
                                                    / v_params->_gearRatios[v_states->_current_gr];

                            // The torque after tranny will then just become as there is no torque converter
                            torque_t = driveTorque(v_params, throttle, v_states->_crankOmega) / v_params->_gearRatios[v_states->_current_gr];

                            if(std::abs((v_states->_u - 0) < 1e-9) && (torque_t < 0)){
                                torque_t = 0;
                            } 

                            ///// Upshift gear for next time step -> Here the crank shaft is directly connected to the gear box
                            if(v_states->_crankOmega > v_params->_upshift_RPS){
                                
                                // check if we have enough gears to upshift
        
                                if(v_states->_current_gr < v_params->_noGears - 1){
                                    v_states->_current_gr++;
                                }
                            }
                            // downshift
                            else if(v_states->_crankOmega < v_params->_downshift_RPS){
                                // check if we can down shift
                                if(v_states->_current_gr > 1){
                                    v_states->_current_gr--;
                                }
                            }                         


                        }

                         //////// Amount of torque transmitted to the wheels
                        // first the front wheels
                        differentialSplit(torque_t, max_bias, tirer_st->_omega, tiref_st->_omega, &tirer_st->_engTor, &tiref_st->_engTor, 1);
                        // then rear wheels


                        // now use this force for our omegas

                        // Get dOmega for each tire
                        double dOmega_f = (1/t_params->_jw) * (tiref_st->_engTor + tiref_st->_My - sgn(tiref_st->_omega) 
                                            * brakeTorque(v_params, brake) - tiref_st->_fx * tiref_st->_rStat);


                        double dOmega_r = (1/t_params->_jw) * (tirer_st->_engTor + tirer_st->_My - sgn(tirer_st->_omega) 
                                            * brakeTorque(v_params, brake) - tirer_st->_fx * tirer_st->_rStat);



                        // integrate omega using the latest dOmega
                        tiref_st->_omega = tiref_st->_omega + t_params->_step * dOmega_f;
                        tirer_st->_omega = tirer_st->_omega + t_params->_step * dOmega_r;
}


__device__ void tmxy_combined(double *f, double *fos, double s, double df0, double sm, double fm, double ss, double fs){
    
    
    double df0loc = 0.0;
    if (sm > 0.0) {
        df0loc = ((2.0 * fm / sm) < df0) ? df0 : (2.0 * fm / sm);
    }

    if (s > 0.0 && df0loc > 0.0) {  // normal operating conditions
        if (s > ss) {               // full sliding
            *f = fs;
            *fos = *f / s;
        } else {
            if (s < sm) {  // adhesion
                double p = df0loc * sm / fm - 2.0;
                double sn = s / sm;
                double dn = 1.0 + (sn + p) * sn;
                *f = df0loc * sm * sn / dn;
                *fos = df0loc / dn;
            } else {
                double a = powf(fm / sm, 2.0) / (df0loc * sm);  // parameter from 2. deriv. of f @ s=sm
                double sstar = sm + (fm - fs) / (a * (ss - sm));    // connecting point
                if (sstar <= ss) {                                  // 2 parabolas
                    if (s <= sstar) {
                        // 1. parabola sm < s < sstar
                        *f = fm - a * (s - sm) * (s - sm);
                    } else {
                        // 2. parabola sstar < s < ss
                        double b = a * (sstar - sm) / (ss - sstar);
                        *f = fs + b * (ss - s) * (ss - s);
                    }
                } else {
                    // cubic fallback function
                    double sn = (s - sm) / (ss - sm);
                    *f = fm - (fm - fs) * sn * sn * (3.0 - 2.0 * sn);
                }
                *fos = *f / s;
            }
        }
    } else {
        *f = 0.0;
        *fos = 0.0;
    }
}






// Advance the tire to the next time step
// update the tire forces which will be used by the vehicle
// whichTire specifies 
// 0 - LF
// 1 - RF
// 2 - LR
// 3 - RR
__device__ void tireAdv(TMeasyState *t_states, const TMeasyParam *t_params, const VehicleState *v_states, const VehicleParam *v_params, const double *controls){
    
    // get the controls and time out
    double t = controls[0];
    //double delta = controls[1] * v_params->_maxSteer;

    double delta = 0;
    if(v_params->_nonLinearSteer){
        // Extract steer map
        MapEntry *steer_map = v_params->_steerMap;
        delta = getMapY(steer_map,controls[1], v_params->_steerMapSize);

    }
    else{
        delta = controls[1] * v_params->_maxSteer;

    }
    double throttle = controls[2];
    double brake = controls[3];

    // Get the whichTire based variables out of the way
    double fz = t_states->_fz; // vertical force 
    double vsy = t_states->_vsy; // y slip velocity
    double vsx = t_states->_vsx; // x slip velocity

    // get our tire deflections so that we can get the loaded radius
    t_states->_xt = fz / t_params->_kt;
    t_states->_rStat = t_params->_r0 - t_states->_xt;


    double r_eff;
    double rdynco;
    if(fz <= t_params->_fzRdynco){
        rdynco = InterpL(fz, t_params->_rdyncoPn, t_params->_rdyncoP2n,t_params->_pn);
        r_eff = rdynco * t_params->_r0 + (1. - rdynco) * t_states->_rStat; 
    }
    else {
        rdynco = t_params->_rdyncoCrit;
        r_eff = rdynco * t_params->_r0 + (1. - rdynco) * t_states->_rStat;  
    }
    // printf("%0.12f, %0.12f\n",t, t_params->_fzRdynco);

    // with this r_eff, we can finalize the x slip velocity
    vsx = vsx - (t_states->_omega * r_eff);

    // get the transport velocity - 0.01 here is to prevent singularity
    double vta = r_eff * abs(t_states->_omega) + 0.01;

    // evaluate the slips
    double sx = -vsx / vta;
    double alpha;
    // only front wheel steering
    alpha = atan2(vsy,vta) - delta;
    double sy = -tan(alpha);

    // limit fz
    if(fz > t_params->_pnmax){
        fz = t_params->_pnmax;
    }

    // calculate all curve parameters through interpolation
    double dfx0 = InterpQ(fz, t_params->_dfx0Pn, t_params->_dfx0P2n, t_params->_pn);
    double dfy0 = InterpQ(fz, t_params->_dfy0Pn, t_params->_dfy0P2n, t_params->_pn);

    double fxm = InterpQ(fz, t_params->_fxmPn, t_params->_fxmP2n, t_params->_pn);
    double fym = InterpQ(fz, t_params->_fymPn, t_params->_fymP2n, t_params->_pn);
    
    double fxs = InterpQ(fz, t_params->_fxsPn, t_params->_fxsP2n, t_params->_pn);
    double fys = InterpQ(fz, t_params->_fysPn, t_params->_fysP2n, t_params->_pn);

    double sxm = InterpL(fz, t_params->_sxmPn, t_params->_sxmP2n, t_params->_pn);
    double sym = InterpL(fz, t_params->_symPn, t_params->_symP2n, t_params->_pn);

    double sxs = InterpL(fz, t_params->_sxsPn, t_params->_sxsP2n, t_params->_pn);
    double sys = InterpL(fz, t_params->_sysPn, t_params->_sysP2n, t_params->_pn);

    // slip normalizing factors
    double hsxn = sxm / (sxm + sym) + (fxm / dfx0) / (fxm / dfx0 + fym / dfy0);
    double hsyn = sym / (sxm + sym) + (fym / dfy0) / (fxm / dfx0 + fym / dfy0);


    // normalized slip
    double sxn = sx / hsxn;
    double syn = sy / hsyn;

    // combined slip
    double sc = hypot(sxn, syn);

    // cos and sine alphs
    double calpha;
    double salpha;
    if(sc > 0){
        calpha = sxn/sc;
        salpha = syn/sc;
    }
    else{
        calpha = sqrt(2.) * 0.5;
        salpha = sqrt(2.) * 0.5;
    }

    // resultant curve parameters in both directions
    double df0 = hypot(dfx0 * calpha * hsxn, dfy0 * salpha * hsyn);
    double fm  = hypot(fxm * calpha, fym * salpha);
    double sm = hypot(sxm * calpha / hsxn, sym * salpha / hsyn);
    double fs = hypot(fxs * calpha, fys * salpha);
    double ss = hypot(sxs * calpha / hsxn, sys * salpha / hsyn);

    // calculate force and force /slip from the curve characteritics
    double f,fos;
    tmxy_combined(&f, &fos, sc, df0, sm, fm, ss, fs);

    // rolling resistance with smoothing
    double vx_min = 0.;
    double vx_max = 0.;


    t_states->_My = -sineStep(vta,vx_min,0.,vx_max,1.) * t_params->_rr * fz * t_states->_rStat * sgn(t_states->_omega);

    
    double h;

    // some normalised slip velocities
    double vtxs = vta * hsxn;
    double vtys = vta * hsyn;


    // some varables needed in the loop
    double fxdyn, fydyn;
    double fxstr, fystr;
    double v_step = v_params->_step;
    double tire_step = t_params->_step;
    // now we integrate to the next vehicle time step
    double tEnd = t + v_step;
    while(t < tEnd){

        // ensure that we integrate exactly to step
        h = min(tire_step, tEnd - t);

        // always integrate using half implicit
        // just a placeholder to simplify the forumlae
        double dFx = -vtxs * t_params->_cx / (vtxs * t_params->_dx + fos);
        
        t_states->_xedot = 1. / (1. - h * dFx) * 
                    (-vtxs * t_params->_cx * t_states->_xe - fos * vsx) /
                    (vtxs * t_params->_dx + fos);

        t_states->_xe = t_states->_xe + h * t_states->_xedot;

        double dFy = -vtys * t_params->_cy / (vtys * t_params->_dy + fos);
        t_states->_yedot = (1. / (1. - h * dFy)) *
                    (-vtys * t_params->_cy * t_states->_ye - fos * (-sy * vta)) /
                    (vtys * t_params->_dy + fos);

        t_states->_ye = t_states->_ye + h * t_states->_yedot;

        // update the force since we need to update the force to get the omegas
        // some wierd stuff happens between the dynamic and structural force
        fxdyn = t_params->_dx * (-vtxs * t_params->_cx * t_states->_xe - fos * vsx) /
                (vtxs * t_params->_dx + fos) + t_params->_cx * t_states->_xe;
        
        fydyn = t_params->_dy * ((-vtys * t_params->_cy * t_states->_ye - fos * (-sy * vta)) /
                (vtys * t_params->_dy + fos)) + (t_params->_cy * t_states->_ye);

        
        fxstr = clamp(t_states->_xe * t_params->_cx + t_states->_xedot * t_params->_dx, -t_params->_fxmP2n, t_params->_fxmP2n);
        fystr = clamp(t_states->_ye * t_params->_cy + t_states->_yedot * t_params->_dy, -t_params->_fymP2n, t_params->_fymP2n);

        double weightx = sineStep(abs(vsx), 1., 1., 1.5, 0.);
        double weighty = sineStep(abs(-sy*vta), 1., 1., 1.5, 0.);

        // now finally get the resultant force
        t_states->_fx = weightx * fxstr + (1.-weightx) * fxdyn;
        t_states->_fy = weighty * fystr + (1.-weighty) * fydyn;

        t += h;
    }

}


__device__ void vehAdv(VehicleState *v_states, const VehicleParam *v_params, const double *fx, const double *fy, const double huf, const double hur){

                double mt = v_params->_m + 2 * (v_params->_muf + v_params->_mur);

    
                v_states->_vdot = -v_states->_u * v_states->_wz + (fy[0] + fy[1])/mt;
                v_states->_udot = v_states->_v * v_states->_wz + (fx[0] + fx[1])/mt;
                v_states->_wzdot = (v_params->_a * fy[0] - v_params->_b * fy[1])/ v_params->_jz;


                v_states->_u = v_states->_u + v_params->_step * v_states->_udot;
                v_states->_v = v_states->_v + v_params->_step * v_states->_vdot;
                v_states->_wz = v_states->_wz + v_params->_step * v_states->_wzdot;

                v_states->_x = v_states->_x + v_params->_step * 
                    (v_states->_u * cos(v_states->_psi) - v_states->_v * sin(v_states->_psi));
                v_states->_y = v_states->_y + v_params->_step * 
                    (v_states->_u * sin(v_states->_psi) + v_states->_v * cos(v_states->_psi));

                v_states->_psi = v_states->_psi + v_params->_step * v_states->_wz;


                // Static vertical load transfer based on d"Almberts principle
                double z1 = (v_params->_m*G*v_params->_b) / (2.*(v_params->_a + v_params->_b)) +
                                (v_params->_muf*G)/2.;
                double z2 = ((v_params->_m*v_params->_h + v_params->_muf*huf + v_params->_mur*hur) *
                                (v_states->_udot - v_states->_wz*v_states->_v)) / (2.*(v_params->_a + v_params->_b));
                
                
                v_states->_fzf = (z1 - z2) > 0. ? (z1 - z2) : 0.;

                double z3 = (v_params->_m*G*v_params->_a) / (2.*(v_params->_a + v_params->_b)) +
                                (v_params->_mur*G)/2.;

                v_states->_fzr = (z3 + z2) > 0. ? (z3 + z2) : 0.;

    
}

// Function that dumps host_response to a csv file for simulation.
// File named according to simulation number and the dump number 
// Dump number gives an idea about what time steps are dumped.
__host__ void dump_csv_all(unsigned int num_vehicles, unsigned int states, double *host_response, unsigned int host_collection_steps, 
                            unsigned int dump_no){


                                for(unsigned int sim_no = 0; sim_no < num_vehicles; sim_no++){
                                    // initialize our csv writers
                                    CSV_writer csv(",");
                                    csv.stream().setf(std::ios::scientific | std::ios::showpos);
                                    csv.stream().precision(8);

                                    csv << "time";
                                    csv << "x";
                                    csv << "y";
                                    csv << "u";
                                    csv << "v";
                                    csv << "psi";    
                                    csv << "wz";
                                    csv << "wf";
                                    csv << "wr";
                                    csv << std::endl;

                                    if(dump_no == 0){ // if this is the first csv dump, then we need to also store inital conditions

                                        csv << 0;
                                        csv << 0;
                                        csv << 0;
                                        csv << 0;
                                        csv << 0;
                                        csv << 0;
                                        csv << 0;    
                                        csv << 0;
                                        csv << 0;
                                        csv << std::endl;                 
                                    }


                                    //at each time step, add it to our csv file

                                    unsigned int collectionStepNo = 0;; // time step counter
                                    while (collectionStepNo < host_collection_steps){
                                        
                                        unsigned int time_offset = collectionStepNo * (states * num_vehicles);

                                        csv << host_response[time_offset + (num_vehicles * 0) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 1) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 2) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 3) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 4) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 5) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 6) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 7) + sim_no];
                                        csv << host_response[time_offset + (num_vehicles * 8) + sim_no];
                                        csv << std::endl;
                                        collectionStepNo++;
                                    }

                                    std::string sno = std::to_string(sim_no);
                                    std::string dno = std::to_string(dump_no); 
                                    
                                    csv.write_to_file("./outs/gpu/test1_veh_" + sno + "_" + dno + ".csv");
                                }


                            }

__host__ void dump_csv_which(unsigned int num_vehicles, unsigned int states, double *host_response, unsigned int host_collection_steps, 
                            unsigned int dump_no, int *which_outs,unsigned int no_outs){

                            for(unsigned int rand_sim = 0; rand_sim < no_outs; rand_sim++){
                                // initialize our csv writers
                                CSV_writer csv(",");
                                csv.stream().setf(std::ios::scientific | std::ios::showpos);
                                csv.stream().precision(8);

                                csv << "time";
                                csv << "x";
                                csv << "y";
                                csv << "u";
                                csv << "v";
                                csv << "psi";    
                                csv << "wz";
                                csv << "wf";
                                csv << "wr";
                                csv << std::endl;


                                if(dump_no == 0){ // if this is the first csv dump, then we need to also store inital conditions

                                    csv << 0;
                                    csv << 0;
                                    csv << 0;
                                    csv << 0;
                                    csv << 0;
                                    csv << 0;
                                    csv << 0;    
                                    csv << 0;
                                    csv << 0;
                                    csv << std::endl;                 
                                }

                                //at each collection step, add it to our csv file

                                unsigned int collectionStepNo = 0;; // time step counter
                                    while (collectionStepNo < host_collection_steps){
                                        
                                        unsigned int time_offset = collectionStepNo * (states * num_vehicles);

                                        csv << host_response[time_offset + (num_vehicles * 0) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 1) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 2) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 3) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 4) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 5) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 6) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 7) + which_outs[rand_sim]];
                                        csv << host_response[time_offset + (num_vehicles * 8) + which_outs[rand_sim]];
                                        csv << std::endl;
                                        collectionStepNo++;
                                    }

                                    std::string sno = std::to_string(which_outs[rand_sim]);
                                    std::string dno = std::to_string(dump_no); 
                                    
                                    csv.write_to_file("./outs/gpu/test1_veh_" + sno + "_" + dno + ".csv");
                                }                            
                            }
