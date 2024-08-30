// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(TestPlatform, Misc)
{
  Int64 page_size = platform::getPageSize();
  std::cout << "PageSize=" << page_size << "\n";
  ASSERT_TRUE(page_size>0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestPlatform, GDBStack)
{
  String str = platform::getGDBStack();
  std::cout << "Stack=" << str << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestPlatform, LLDBStack)
{
  String str = platform::getLLDBStack();
  std::cout << "Stack=" << str << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
