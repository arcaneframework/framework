// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/internal/MemoryPool.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::impl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MyMemoryPoolAllocator
: public IMemoryPoolAllocator
{
  void* allocateMemory(size_t size) override { return std::malloc(size); }
  void freeMemory(void* address,size_t) override { std::free(address); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(MemoryPool, Misc)
{
  MyMemoryPoolAllocator my_allocator;
  MemoryPool memory_pool(&my_allocator);

  void* a1 = memory_pool.allocateMemory(25);
  void* a2 = memory_pool.allocateMemory(47);
  memory_pool.freeMemory(a1, 25);
  memory_pool.freeMemory(a2, 47);
  memory_pool.dumpStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
