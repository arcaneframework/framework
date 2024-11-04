﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/internal/MemoryPool.h"
#include "arcane/utils/String.h"

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
  MemoryPool memory_pool(&my_allocator, "MyMemoryPool");
  size_t max_block_size = 1024 * 8;
  memory_pool.setMaxCachedBlockSize(1024 * 8);

  void* a1 = memory_pool.allocateMemory(25);
  void* a2 = memory_pool.allocateMemory(47);
  memory_pool.freeMemory(a1, 25);
  memory_pool.freeMemory(a2, 47);
  void* a3 = memory_pool.allocateMemory(25);
  memory_pool.dumpStats(std::cout);
  memory_pool.dumpFreeMap(std::cout);
  memory_pool.freeMemory(a3, 25);

  void* a4 = memory_pool.allocateMemory(max_block_size * 2);
  memory_pool.freeMemory(a4, max_block_size * 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
