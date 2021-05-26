//
// Created by jared on 5/26/2021.
//

#include <gtest/gtest.h>
#include "emulator.h"

// Test if emulator initialization works
TEST(HelloTest, BasicAssertions) {

EXPECT_STRNE("hello", "world");
Emulator<int, int> emulator("./test");
EXPECT_EQ(emulator.i, 21);
}