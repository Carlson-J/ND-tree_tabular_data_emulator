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
#define ND_TREE_EMULATOR_TYPE unsigned char, 1, 1, 13, 32
#define ND_TREE_EMULATOR_NAME_SETUP non_linear_multi_model_emulator_setup
#define ND_TREE_EMULATOR_NAME_INTERPOLATE non_linear_multi_model_emulator_interpolate
#define ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE non_linear_multi_model_emulator_interpolate_single
#define ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE_DX1 non_linear_multi_model_emulator_interpolate_single_dx1
#define ND_TREE_EMULATOR_NAME_FREE non_linear_multi_model_emulator_free
#define POINT_INPUTS double* x0
#define POINT_INPUTS_SINGLE double& x0
#define POINT_GROUPING double* points[1] = {x0}
#define POINT_GROUPING_SINGLE double points[1] = {x0}
#define POINT_ARG x0
