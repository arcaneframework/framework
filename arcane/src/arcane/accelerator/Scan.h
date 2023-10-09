// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Scan.h                                                      (C) 2000-2023 */
/*                                                                           */
/* Gestion des opérations de scan pour les accélérateurs.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_SCAN_H
#define ARCANE_ACCELERATOR_SCAN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#if defined(ARCANE_COMPILING_CUDA)
#include "arcane/accelerator/cuda/CudaAccelerator.h"
#include <cub/cub.cuh>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour effectuer une réduction 'min'.
 */
template <typename DataType>
class ScannerSum
{
 public:

  explicit ScannerSum(RunQueue* queue)
  : m_queue(queue)
  {}

 public:

  void exclusiveSum(SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());
    const DataType* input_data = input.data();
    DataType* output_data = output.data();
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    if (m_queue)
      exec_policy = m_queue->executionPolicy();
    switch (exec_policy) {
    case eExecutionPolicy::CUDA:
#if defined(ARCANE_COMPILING_CUDA)
    {
      void* d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      ARCANE_CHECK_CUDA(::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_data, nb_item));
      // Allocate temporary storage for inclusive prefix sum
      ARCANE_CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      ARCANE_CHECK_CUDA(::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_data, nb_item));
      ARCANE_CHECK_CUDA(::cudaFree(d_temp_storage));
    } break;
#else
      ARCANE_FATAL("Requesting CUDA kernel execution but the kernel is not compiled with CUDA."
                   " You need to compile the file containing this kernel with CUDA compiler.");
#endif
    case eExecutionPolicy::HIP:
      ARCANE_FATAL("Not yet implemented");
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
    case eExecutionPolicy::Sequential: {
      DataType sum = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        output[i] = sum;
        sum += input[i];
      }
    } break;
    default:
      ARCANE_FATAL("Invalid execution policy '{0}'", exec_policy);
    }
  }

 private:

  RunQueue* m_queue = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
