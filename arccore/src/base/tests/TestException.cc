// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/IndexOutOfRangeException.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotImplementedException.h"

#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{

void _doCheckRange(Int64 i, Int64 min_inclusive, Int64 max_exclusive)
{
  arccoreCheckRange(i, min_inclusive, max_exclusive);
}
void _doCheckRange(Int64 i, Int64 max_exclusive)
{
  arccoreCheckAt(i, max_exclusive);
}

} // namespace

TEST(Exception,Range)
{
  EXPECT_THROW(_doCheckRange(0,0,0),IndexOutOfRangeException);
  EXPECT_THROW(_doCheckRange(-2,-2,-2),IndexOutOfRangeException);
  EXPECT_THROW(_doCheckRange(-3,-1,5),IndexOutOfRangeException);
  EXPECT_THROW(_doCheckRange(15,-2,10),IndexOutOfRangeException);
  EXPECT_THROW(_doCheckRange(15,4,15),IndexOutOfRangeException);
  EXPECT_NO_THROW(_doCheckRange(4,4,15));
  EXPECT_NO_THROW(_doCheckRange(-4,-4,5));
  EXPECT_NO_THROW(_doCheckRange(15,-3,20));

  EXPECT_THROW(_doCheckRange(15,15),IndexOutOfRangeException);
  EXPECT_THROW(_doCheckRange(-3,15),IndexOutOfRangeException);
  EXPECT_THROW(_doCheckRange(-3,0),IndexOutOfRangeException);
  EXPECT_THROW(_doCheckRange(0,0),IndexOutOfRangeException);
  EXPECT_NO_THROW(_doCheckRange(15,17));

  ARCCORE_CHECK_RANGE(3,-2,19);
}

TEST(Exception, ArccoreThrow)
{
  EXPECT_THROW(ARCCORE_FATAL("Test"), FatalErrorException);
  EXPECT_THROW(ARCCORE_FATAL_IF(true, "Test"), FatalErrorException);
  EXPECT_NO_THROW(ARCCORE_FATAL_IF(false, "Test"));

  EXPECT_THROW(ARCCORE_THROW(NotImplementedException, "Test"), NotImplementedException);
  EXPECT_THROW(ARCCORE_THROW_IF(true, NotImplementedException, "Test"), NotImplementedException);
  EXPECT_NO_THROW(ARCCORE_THROW_IF(false, NotImplementedException, "Test"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
