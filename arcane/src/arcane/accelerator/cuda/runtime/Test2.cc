// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Test2.cpp                                                   (C) 2000-2025 */
/*                                                                           */
/* Fichier contenant les tests pour l'implémentation CUDA.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/NumArray.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/Item.h"
#include "arcane/core/MathUtils.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/cuda/CudaAccelerator.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

using namespace Arccore;
using namespace Arcane;
using namespace Arcane::Accelerator;

__global__ void MyVecAdd3(double* a, double* b, double* out, int nb_value)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  cg::grid_group this_grid_group = cg::this_grid();
  if (i >= nb_value)
    return;
  this_grid_group.sync();
  out[i] = a[i] + b[i];
  if (i < 10) {
    printf("A=%d %lf %lf %lf grid_size=%llu \n", i, a[i], b[i], out[i], this_grid_group.size());
  }
}

extern "C" void arcaneTestCooperativeLaunch()
{
  std::cout << "Test Cooperative Launch\n";
  constexpr int vsize = 2000;
  std::vector<double> a(vsize);
  std::vector<double> b(vsize);
  std::vector<double> out(vsize);
  for (size_t i = 0; i < vsize; ++i) {
    a[i] = (double)(i + 1);
    b[i] = (double)(i * i + 1);
    out[i] = 0.0; //a[i] + b[i];
  }
  size_t mem_size = vsize * sizeof(double);
  double* d_a = nullptr;
  cudaMalloc(&d_a, mem_size);
  double* d_b = nullptr;
  cudaMalloc(&d_b, mem_size);
  double* d_out = nullptr;
  cudaMalloc(&d_out, mem_size);

  cudaMemcpy(d_a, a.data(), mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), mem_size, cudaMemcpyHostToDevice);
  int threadsPerBlock = 256;
  int blocksPerGrid = (vsize + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "CALLING kernel tpb=" << threadsPerBlock << " bpg=" << blocksPerGrid << "\n";
  int nb_value = vsize;
  void* args[] = { &d_a, &d_b, &d_out, &nb_value };
  const void* func_ptr = reinterpret_cast<const void*>(&MyVecAdd3);
  ARCANE_CHECK_CUDA(cudaLaunchCooperativeKernel(func_ptr, dim3(blocksPerGrid), dim3(threadsPerBlock), args, 0, 0));
  ARCANE_CHECK_CUDA(cudaDeviceSynchronize());
  ARCANE_CHECK_CUDA(cudaMemcpy(out.data(), d_out, mem_size, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < 10; ++i)
    std::cout << "V=" << out[i] << "\n";
}
