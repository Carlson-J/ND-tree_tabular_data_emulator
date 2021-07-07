#undef ND_TREE_EMULATOR_TYPE
#undef ND_TREE_EMULATOR_NAME_SETUP
#undef ND_TREE_EMULATOR_NAME_INTERPOLATE
#undef ND_TREE_EMULATOR_NAME_FREE
#define ND_TREE_EMULATOR_TYPE unsigned short, unsigned char, 1, 2, 12, 96
#define ND_TREE_EMULATOR_NAME_SETUP miss_aligned_2d_extended_emulator_setup
#define ND_TREE_EMULATOR_NAME_INTERPOLATE miss_aligned_2d_extended_emulator_interpolate
#define ND_TREE_EMULATOR_NAME_FREE miss_aligned_2d_extended_emulator_free
