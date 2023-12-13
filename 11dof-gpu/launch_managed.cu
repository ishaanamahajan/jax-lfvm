#include<cuda.h>
#include<iostream>
#include<random>
#include<cuda_runtime.h>
#include<math.h>
#include "rom.cuh"
// #include "csvwriter.h"



/*
This function takes command line inputs for the number of vehicles to be launched.
We also take the threads per block argument for optimization purposes. From the threads per block
the number of blocks will interanlly be calculated so that threads per block * blocks will be 
equal to the number of vehicles to be launched.
*/
int main(int argc, char *argv[]){

    // get the necessray inputs num_vehicles and threads per block
    char *pCh; 
    // input n - number of rows and columns
    unsigned int num_vehicles = strtoul(argv[1], &pCh, 10);
    // input threads per block
    const unsigned int threads_per_block = strtoul(argv[2], &pCh, 10);


    
    std::string fileName = "./inputs/test_set2.txt"; // Controls file name
    char *vehParamsJSON = (char *)"./jsons/HMMWV.json"; // HMMWV vehicle parameters
    char *tireParamsJSON = (char *)"./jsons/TMeasy.json"; // Tire parameters


    /* Initialize an array of "vehicle" structures. 
       These vehicle structures will contain the tire parameters,
       the vehicle parameters and the controls (driver data) for the vehicle.
       This will be passed to the GPU. The length of this array is thus num_vehilces
    */
    // we will just set these structures with values from our JSON files
    VehicleParam veh_param;
    TMeasyParam tire_param;

    // States
    VehicleState veh1_st;
    TMeasyState tiref_st;
    TMeasyState tirer_st;
    
    
    // Initialize our simData using managed memory
    SimData *simData;
    CHECK_CUDA_ERROR(cudaMallocManaged((void **)&simData, sizeof(SimData) * num_vehicles));

    // Initialize simState using managed memory

    SimState *simState;
    CHECK_CUDA_ERROR(cudaMallocManaged((void **)&simState, sizeof(SimState) * num_vehicles));

    // Fill up this array of simData's with initialized vehicle and tire parameters from JSON files
    for(size_t i =0; i < num_vehicles; i++){
        // set the values - for now, all our vehicles are exactly the same
        setVehParamsJSON(veh_param,vehParamsJSON);
        setTireParamsJSON(tire_param,tireParamsJSON);
        vehInit(&veh1_st, &veh_param);
        tireInit(&tire_param);

        // read driver data into a vector 
        std::vector<Entry> driverData;
        driverInput(driverData, fileName);

        unsigned int len = driverData.size();
        
        // allocate memory based on the length of the vector
        CHECK_CUDA_ERROR(cudaMallocManaged((void **)&simData[i]._driver_data, sizeof(Entry) * len));
        
        // Now fill up Simulation Data
        simData[i]._veh_param = veh_param;
        simData[i]._tire_param = tire_param;
        std::copy(driverData.begin(),driverData.end(), simData[i]._driver_data);
        simData[i]._driver_data_len = len;

        // Fill up simState
        simState[i]._veh_st = veh1_st;
        simState[i]._tiref_state = tiref_st;
        simState[i]._tirer_state = tirer_st;
        
    }

    cudaMemPrefetchAsync(&simData, sizeof(SimData) * num_vehicles, 0); // move the simData onto the GPU
    cudaMemPrefetchAsync(&simState, sizeof(SimState) * num_vehicles, 0); // move the simData onto the GPU


    //////////////////////////////   Write the response vector for each vehicle into seperate csv files ////////////////////////////////////////////////
    bool data_output = 1;
    // Flag for if we want to store all the outputs or only some in random - 0 means we don't store all
    bool store_all = 0; 

    /// We will need a list of random vehicles to save as CSV  if we don't want to save all

    // Write "no_outs" outputs - These are simulations picked at random
    int no_outs = min(50,num_vehicles);
    // fill a vector with "no_outs" random numbers between 0 and num_vehicles
    float some_seed = 68;
    std::mt19937 generator(some_seed);

    // random floats between -1 and 1
    const int minval_input = 0., maxval_input = num_vehicles;
    std::uniform_int_distribution<int> dist_input(minval_input, maxval_input);

    int *which_outs = new int [no_outs];


    // fill in our numbers
    for(std::size_t i = 0; i < no_outs; i++ ){
        which_outs[i] = dist_input(generator);
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////// Bunch of code for running small simulations and flushing the response //////////////////////////////////////


    // Launch the kernel every "device_time" -> This is the amount of space we need
    // our device response to be
    double device_time = 2.000;

    double step = 0.001;
    unsigned int collection_rate = 10; // collect data every time step
    unsigned int device_timeSteps = floor(device_time/step); // Number of timeSteps in each kernel launch
    unsigned int device_collection_steps = (device_timeSteps / collection_rate);
    int states = 9;    // Number of states that need to be stored
    unsigned int device_size = sizeof(double) * num_vehicles * states * (device_collection_steps + 1);
    double *device_response;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&device_response, device_size));



    


    // Now we have a host response array that we will flush into a csv file ever "host_time" seconds
    // This defines how much space we need

    double host_time = 10.; // for now we set this to 10 secs
    unsigned int host_timeSteps = floor(host_time/step);
    unsigned int host_collection_steps = (host_timeSteps / collection_rate);
    unsigned int csv_dump_steps = (int) host_time / (int) (device_time); // after "csv_dump_steps" many kernel steps do we need to dump to a csv
    // assign the required memory using new
    double *host_response = new double [num_vehicles * states * host_collection_steps](); // () apparently assigns 0's




    
    // define simulation duration
    double endTime = 20.;
    // calculate number of blocks required
    const unsigned int blocks = (num_vehicles + threads_per_block - 1)/threads_per_block;


    ///////////////////////////////////////////// Start the simulation /////////////////////////////////////////////////////////////////////////////////

    // // For loop that launches our kernel every "device_time" seconds
    double current_time = 0.;
    unsigned int kernel_step = 0;
    unsigned int steps_post_host_time = kernel_step; // These are the steps that are reset to 0 every "host_time" seconds
    unsigned int csv_dumps = 0; // counts the number of csv dumps
    
    // call kernel with some timing code around
    cudaEvent_t startEvent1, stopEvent1; 
    cudaEventCreate(&startEvent1); 
    cudaEventCreate(&stopEvent1);
    cudaEventRecord(startEvent1, 0);

    while(current_time < endTime){


        
        double kernel_endTime = current_time + device_time; // the time at which the kernel will return
        
        // Lanuch the kernel
        run<<<blocks, threads_per_block>>>(simData, device_response, current_time, kernel_endTime, step, collection_rate, states, num_vehicles,simState); 

        
        current_time = kernel_endTime; // We have reached the kernel end time so update current time

        // copy device to response
        unsigned int response_filled = num_vehicles * states * device_collection_steps * steps_post_host_time;
        CHECK_CUDA_ERROR(cudaMemcpy(host_response + response_filled, device_response, sizeof(double) * num_vehicles * states * device_collection_steps, cudaMemcpyDeviceToHost));

        // Incremenet the kernel step here
        kernel_step++;
        steps_post_host_time++;


        // syncronize after each run since we have to move data around and to dump in csv 
        // we have to ensure that we have all the data in host response
        cudaDeviceSynchronize();


        // Now if host_response is full, i.e. if its been "host_time"
        // then dump host_response to csv and reset it to 0
    
        if((kernel_step % csv_dump_steps == 0) && (kernel_step != 0)){
            if(data_output){

                if(store_all){ // if we need to store all, then we call a different csv write function that does not require which_outs
                    dump_csv_all(num_vehicles, states, host_response, host_collection_steps, csv_dumps);
                    
                }
                else{
                    dump_csv_which(num_vehicles, states, host_response, host_collection_steps, csv_dumps, which_outs, no_outs);
                }
                csv_dumps++;
                steps_post_host_time = 0;
            }
            else{ // if we don't have to dump to a csv, just reset steps_post_host_time
                steps_post_host_time = 0;
            }
        }

    }

    cudaEventRecord(stopEvent1, 0); 
    cudaEventSynchronize(stopEvent1);

    float elapsedTime1;
    cudaEventElapsedTime(&elapsedTime1, startEvent1, stopEvent1);
    std::cout<<elapsedTime1<< "\n";
    cudaEventDestroy(startEvent1); 
    cudaEventDestroy(stopEvent1);


    // Free responses
   
    cudaFree(host_response);
    cudaFree(device_response);
    // delete [] host_response;

    // Free the managed memory for each driver data
    for(size_t i =0; i < num_vehicles; i++){
        cudaFree(simData[i]._driver_data);
    }
    
    cudaFree(simData);
    cudaFree(simState);





}

