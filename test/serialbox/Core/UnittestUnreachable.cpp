//===-- serialbox/Core/UnittestUnreachable.cpp --------------------------------------*- C++ -*-===//
//
//                                    S E R I A L B O X
//
// This file is distributed under terms of BSD license.
// See LICENSE.txt for more information
//
//===------------------------------------------------------------------------------------------===//
//
/// \file
/// This file contains the unittests for the Unreachable utility.
///
//===------------------------------------------------------------------------------------------===//

#include "serialbox/Core/Type.h"
#include "serialbox/Core/Unreachable.h"
#include <gtest/gtest.h>

using namespace serialbox;

TEST(UnreachableTest, Unreachable) {
#if defined(SERIALBOX_RUN_DEATH_TESTS) && !defined(NDEBUG)
  ASSERT_DEATH_IF_SUPPORTED(serialbox_unreachable("blub"), "^FATAL ERROR: UNREACHABLE executed");
#endif
}
