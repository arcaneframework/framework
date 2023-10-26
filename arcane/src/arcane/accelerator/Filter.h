// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Filtering.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Algorithme de filtrage.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_FILTERING_H
#define ARCANE_ACCELERATOR_FILTERING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/CommonUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer un filtrage
 *
 * \a DataType est le type de donnée.
 * \a FlagType est le type du tableau de filtre.
 */
template <typename DataType, typename FlagType>
class GenericFiltering
{
  // TODO: Faire le malloc sur le device associé à la queue.
  //       et aussi regarder si on peut utiliser mallocAsync().

 public:

  explicit GenericFiltering(RunQueue* queue)
  : m_queue(queue)
  {}

 public:

  void apply(SmallSpan<const DataType> input, SmallSpan<DataType> output, SmallSpan<FlagType> flag)
  {
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());
    [[maybe_unused]] const DataType* input_data = input.data();
    [[maybe_unused]] DataType* output_data = output.data();
    [[maybe_unused]] const FlagType* flag_data = flag.data();
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    if (m_queue)
      exec_policy = m_queue->executionPolicy();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA:
    {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(m_queue);
      // Premier appel pour connaitre la taille pour l'allocation
      int* nb_out_ptr = nullptr;
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::Flagged(nullptr, temp_storage_size,
                                                     input_data, flag_data, output_data, nb_out_ptr, nb_item, stream));

      m_algo_storage.allocate(temp_storage_size);
      m_device_nb_out_storage.allocate();
      nb_out_ptr = m_device_nb_out_storage.address();
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::Flagged(m_algo_storage.address(), temp_storage_size,
                                                     input_data, flag_data, output_data, nb_out_ptr, nb_item, stream));
      ARCANE_CHECK_CUDA(::cudaMemcpyAsync(&m_host_nb_out, nb_out_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP:
    {
      size_t temp_storage_size = 0;
      // Premier appel pour connaitre la taille pour l'allocation
      hipStream_t stream = impl::HipUtils::toNativeStream(m_queue);
      int* nb_out_ptr = nullptr;
      ARCANE_CHECK_HIP(rocprim::select(nullptr, temp_storage_size, input_data, flag_data, output_data,
                                       nb_out_ptr, nb_item, stream));

      m_algo_storage.allocate(temp_storage_size);
      m_device_nb_out_storage.allocate();
      nb_out_ptr = m_device_nb_out_storage.address();

      ARCANE_CHECK_HIP(rocprim::select(m_algo_storage.address(), temp_storage_size, input_data, flag_data, output_data,
                                       nb_out_ptr, nb_item, stream));
      ARCANE_CHECK_HIP(::hipMemcpyAsync(&m_host_nb_out, nb_out_ptr, sizeof(int), hipMemcpyDeviceToHost, stream));
    }
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      Int32 index = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        if (flag[i] != 0) {
          output[index] = input[i];
          ++index;
        }
      }
      m_host_nb_out = index;
    } break;
    default:
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
  }

 public:

  //! Nombre d'éléments en sortie. Il faut être certain d'avoir synchronisé la RunQueue
  Int32 nbOutputElement() const
  {
    return m_host_nb_out;
  }

 private:

  RunQueue* m_queue = nullptr;
  GenericDeviceStorage m_algo_storage;
  DeviceStorage<int> m_device_nb_out_storage;
  int m_host_nb_out = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
