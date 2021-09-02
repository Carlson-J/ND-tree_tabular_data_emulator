# README for the *non_linear_multi_model* emulator
    This folder should contain three file, 
        *non_linear_multi_model_table.hdf5*: Contains the data for the compact emulator
        *non_linear_multi_model_cpp_params.h*: Contains the #define's used when creating the C++ lib
        *non_linear_multi_model_lib.so*: C++ lib that has C extern functions that can be called to make, use, and destroy the
            emulator. 

    The function names in *non_linear_multi_model_lib.so* that can be called are named based on the emulators name and are as follows:
        *non_linear_multi_model_emulator_setup*: Constructs an emulator C++ object.
        *non_linear_multi_model_emulator_interpolate*: Calls the emulator object for interpolation.
        *non_linear_multi_model_emulator_free*: Frees the memory allocated by *non_linear_multi_model_emulator_setup*.

    