// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Sorter.h                                                    (C) 2000-2024 */
/*                                                                           */
/* Algorithme de tri.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_SORTER_H
#define ARCANE_ACCELERATOR_SORTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NumArray.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/CommonUtils.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base pour effectuer un tri.
 *
 * Contient les arguments nécessaires pour effectuer le tri.
 */
class ARCANE_ACCELERATOR_EXPORT GenericSorterBase
{
  friend class GenericSorterMergeSort;

 public:

  GenericSorterBase(const RunQueue& queue);

 protected:

  RunQueue m_queue;
  GenericDeviceStorage m_algo_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer le tri d'une liste.
 *
 * La classe utilisateur associée est GenericSorter
 */
class GenericSorterMergeSort
{
  // TODO: Faire le malloc sur le device associé à la queue.
  //       et aussi regarder si on peut utiliser mallocAsync().

 public:

  template <typename InputIterator, typename OutputIterator>
  void apply(GenericSorterBase& s, Int32 nb_item, InputIterator input_iter, OutputIterator output_iter)
  {
    RunQueue queue = s.m_queue;
    eExecutionPolicy exec_policy = queue.executionPolicy();
    using KeyType = typename InputIterator::value_type;
    auto select_lambda = [] ARCCORE_HOST_DEVICE(const KeyType& a, const KeyType& b) {
      return a < b;
    };
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(&queue);
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_CUDA(::cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_storage_size,
                                                             input_iter, output_iter, nb_item,
                                                             select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      ARCANE_CHECK_CUDA(::cub::DeviceMergeSort::SortKeysCopy(s.m_algo_storage.address(), temp_storage_size,
                                                             input_iter, output_iter, nb_item,
                                                             select_lambda, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      hipStream_t stream = impl::HipUtils::toNativeStream(&queue);
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_HIP(rocprim::merge_sort(nullptr, temp_storage_size, input_iter, output_iter,
                                           nb_item, select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);

      ARCANE_CHECK_HIP(rocprim::merge_sort(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                           nb_item, select_lambda, stream));
    } break;
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      // Copie input dans output
      auto output_iter_begin = output_iter;
      for (Int32 i = 0; i < nb_item; ++i) {
        *output_iter = *input_iter;
        ++output_iter;
        ++input_iter;
      }
      std::sort(output_iter_begin, output_iter, select_lambda);
    } break;
    default:
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithme générique de tri sur accélérateur.
 *
 * \warning API en cours de développement. Ne pas utiliser en dehors d'Arcane
 */
class GenericSorter
: private impl::GenericSorterBase
{
 public:

  explicit GenericSorter(const RunQueue& queue)
  : impl::GenericSorterBase(queue)
  {
  }

 public:

  template <typename InputIterator, typename OutputIterator>
  void apply(Int32 nb_item, InputIterator input_iter, OutputIterator output_iter)
  {
    impl::GenericSorterBase* base_ptr = this;
    impl::GenericSorterMergeSort gf;
    gf.apply(*base_ptr, nb_item, input_iter, output_iter);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
