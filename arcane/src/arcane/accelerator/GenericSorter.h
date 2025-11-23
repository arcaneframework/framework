// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericSorter.h                                             (C) 2000-2025 */
/*                                                                           */
/* Algorithme de tri.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_GENERICSORTER_H
#define ARCANE_ACCELERATOR_GENERICSORTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NumArray.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/CommonUtils.h"

#if defined(ARCANE_COMPILING_SYCL)
#include "arcane/accelerator/RunCommandLoop.h"
#endif

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

  explicit GenericSorterBase(const RunQueue& queue);

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

  template <typename CompareLambda, typename InputIterator, typename OutputIterator>
  void apply(GenericSorterBase& s, Int32 nb_item, InputIterator input_iter,
             OutputIterator output_iter, const CompareLambda& compare_lambda)
  {
    RunQueue queue = s.m_queue;
    eExecutionPolicy exec_policy = queue.executionPolicy();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = Impl::CudaUtils::toNativeStream(&queue);
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_CUDA(::cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_storage_size,
                                                             input_iter, output_iter, nb_item,
                                                             compare_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      ARCANE_CHECK_CUDA(::cub::DeviceMergeSort::SortKeysCopy(s.m_algo_storage.address(), temp_storage_size,
                                                             input_iter, output_iter, nb_item,
                                                             compare_lambda, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      hipStream_t stream = Impl::HipUtils::toNativeStream(&queue);
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_HIP(rocprim::merge_sort(nullptr, temp_storage_size, input_iter, output_iter,
                                           nb_item, compare_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);

      ARCANE_CHECK_HIP(rocprim::merge_sort(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                           nb_item, compare_lambda, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      {
        // Copie input dans output
        auto command = makeCommand(queue);
        command << RUNCOMMAND_LOOP1(iter, nb_item)
        {
          auto [i] = iter();
          *(output_iter + i) = *(input_iter + i);
        };
      }
#if defined(ARCANE_HAS_ONEDPL)
      sycl::queue true_queue = AcceleratorUtils::toSyclNativeStream(queue);
      auto policy = oneapi::dpl::execution::make_device_policy(true_queue);
      oneapi::dpl::sort(policy, output_iter, output_iter + nb_item, compare_lambda);
#elif defined(__ADAPTIVECPP__)
      sycl::queue true_queue = AcceleratorUtils::toSyclNativeStream(queue);
      sycl::event e = acpp::algorithms::sort(true_queue, output_iter, output_iter + nb_item, compare_lambda);
      e.wait();
#else
      ARCANE_THROW(NotImplementedException, "Sort is only implemented for SYCL back-end using oneDPL or AdaptiveCpp");
#endif
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
      std::sort(output_iter_begin, output_iter, compare_lambda);
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

  /*!
   * \brief Tri les entités.
   *
   * Remplit \a output avec les valeurs de \a input triées via le comparateur
   * par défaut pour le type \a DataType. Le tableau \a input n'est pas modifié.
   *
   * \pre output.size() >= input.size()
   */
  template <typename DataType>
  void apply(SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    impl::GenericSorterBase* base_ptr = this;
    impl::GenericSorterMergeSort gf;
    Int32 nb_item = input.size();
    if (output.size() < nb_item)
      ARCANE_FATAL("Output size '{0}' is smaller than input size '{1}'",
                   output.size(), nb_item);
    gf.apply(*base_ptr, nb_item, input.data(), output.data(), std::less<DataType>{});
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
