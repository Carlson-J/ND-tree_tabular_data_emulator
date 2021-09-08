# README for the *miss_aligned_2d_extended* emulator
    This folder should contain three file, 
        *miss_aligned_2d_extended_table.hdf5*: Contains the data for the compact emulator
        *miss_aligned_2d_extended_cpp_params.h*: Contains the #define's used when creating the C++ lib
        *miss_aligned_2d_extended_lib.so*: C++ lib that has C extern functions that can be called to make, use, and destroy the
            emulator. 

    The function names in *miss_aligned_2d_extended_lib.so* that can be called are named based on the emulators name and are as follows:
        *miss_aligned_2d_extended_emulator_setup*: Constructs an emulator C++ object.
        *miss_aligned_2d_extended_emulator_interpolate*: Calls the emulator object for interpolation.
        *miss_aligned_2d_extended_emulator_free*: Frees the memory allocated by *miss_aligned_2d_extended_emulator_setup*.

    