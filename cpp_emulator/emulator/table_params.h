#undef ND_TREE_EMULATOR_TYPE
#undef ND_TREE_EMULATOR_NAME_SETUP
#undef ND_TREE_EMULATOR_NAME_INTERPOLATE
#undef ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE
#undef ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE_DX1
#undef ND_TREE_EMULATOR_NAME_FREE
#undef POINT_INPUTS
#undef POINT_INPUTS_SINGLE
#undef POINT_GROUPING
#undef POINT_GROUPING_SINGLE
#undef POINT_ARG
#define ND_TREE_EMULATOR_TYPE unsigned char, 1, 2, 23, 102
#define ND_TREE_EMULATOR_NAME_SETUP miss_aligned_2d_extended_emulator_setup
#define ND_TREE_EMULATOR_NAME_INTERPOLATE miss_aligned_2d_extended_emulator_interpolate
#define ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE miss_aligned_2d_extended_emulator_interpolate_single
#define ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE_DX1 miss_aligned_2d_extended_emulator_interpolate_single_dx1
#define ND_TREE_EMULATOR_NAME_FREE miss_aligned_2d_extended_emulator_free
#define POINT_INPUTS double* x0, double* x1
#define POINT_INPUTS_SINGLE double& x0, double& x1
#define POINT_GROUPING double* points[2] = {x0,x1}
#define POINT_GROUPING_SINGLE double points[2] = {x0,x1}
#define POINT_ARG x0,x1 
