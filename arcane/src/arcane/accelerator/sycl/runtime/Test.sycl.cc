// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Test.sycl.cc                                                (C) 2000-2024 */
/*                                                                           */
/* Fichier contenant les tests pour l'implémentation SYCL.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/sycl/SyclAccelerator.h"

#include "arcane/utils/NumArray.h"

using namespace Arccore;
using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Test Appel pure SYCL
extern "C" int arcaneTestSycl1()
{
  const int N = 8;
  std::cout << "TEST1\n";

  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  int* data = sycl::malloc_shared<int>(N, q);

  for (int i = 0; i < N; i++)
    data[i] = i;

  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
     data[i] *= 2;
   })
  .wait();

  for (int i = 0; i < N; i++)
    std::cout << data[i] << std::endl;
  sycl::free(data, q);

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Idem Test1 avec des NumArray
extern "C" int arcaneTestSycl2()
{
  const int N = 8;
  std::cout << "TEST 2\n";

  sycl::queue q;

  NumArray<Int32, MDDim1> data(N);

  for (int i = 0; i < N; i++)
    data[i] = i;

  Span<Int32> inout_data(data.to1DSpan());
  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
     inout_data[i] *= 3;
   })
  .wait();

  for (int i = 0; i < N; i++)
    std::cout << data[i] << std::endl;

  return 0;
}
