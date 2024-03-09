// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Scan.h                                                      (C) 2000-2024 */
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
#include "arcane/accelerator/CommonUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateur de Scan pour les sommes
template <typename DataType>
class ScannerSumOperator
{
 public:

  constexpr ARCCORE_HOST_DEVICE DataType operator()(const DataType& a, const DataType& b) const
  {
    return a + b;
  }
  static DataType initialValue() { return {}; }
};

//! Opérateur de Scan pour le minimum
template <typename DataType>
class ScannerMinOperator
{
 public:

  constexpr ARCCORE_HOST_DEVICE DataType operator()(const DataType& a, const DataType& b) const
  {
    return (a < b) ? a : b;
  }
  static DataType initialValue() { return std::numeric_limits<DataType>::max(); }
};

//! Opérateur de Scan pour le maximum
template <typename DataType>
class ScannerMaxOperator
{
 public:

  constexpr ARCCORE_HOST_DEVICE DataType operator()(const DataType& a, const DataType& b) const
  {
    return (a < b) ? b : a;
  }
  static DataType initialValue() { return std::numeric_limits<DataType>::lowest(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer un scan exlusif ou inclusif avec un opérateur spécifique.
 */
class ScannerImpl
{
  // TODO: Faire le malloc sur le device associé à la queue.
  //       et aussi regarder si on peut utiliser mallocAsync().

 public:

  explicit ScannerImpl(const RunQueue& queue)
  : m_queue(queue)
  {}

 public:

  template <bool IsExclusive, typename InputIterator, typename OutputIterator, typename Operator, typename DataType>
  void apply(Int32 nb_item, InputIterator input_data, OutputIterator output_data, DataType init_value, Operator op)
  {
    eExecutionPolicy exec_policy = m_queue.executionPolicy();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(&m_queue);
      // Premier appel pour connaitre la taille pour l'allocation
      if constexpr (IsExclusive)
        ARCANE_CHECK_CUDA(::cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_size,
                                                           input_data, output_data, op, init_value, nb_item, stream));
      else
        ARCANE_CHECK_CUDA(::cub::DeviceScan::InclusiveScan(nullptr, temp_storage_size,
                                                           input_data, output_data, op, nb_item, stream));
      void* temp_storage = m_storage.allocate(temp_storage_size);
      if constexpr (IsExclusive)
        ARCANE_CHECK_CUDA(::cub::DeviceScan::ExclusiveScan(temp_storage, temp_storage_size,
                                                           input_data, output_data, op, init_value, nb_item, stream));
      else
        ARCANE_CHECK_CUDA(::cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_size,
                                                           input_data, output_data, op, nb_item, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // Premier appel pour connaitre la taille pour l'allocation
      hipStream_t stream = impl::HipUtils::toNativeStream(&m_queue);
      if constexpr (IsExclusive)
        ARCANE_CHECK_HIP(rocprim::exclusive_scan(nullptr, temp_storage_size, input_data, output_data,
                                                 init_value, nb_item, op, stream));
      else
        ARCANE_CHECK_HIP(rocprim::inclusive_scan(nullptr, temp_storage_size, input_data, output_data,
                                                 nb_item, op, stream));
      void* temp_storage = m_storage.allocate(temp_storage_size);
      if constexpr (IsExclusive)
        ARCANE_CHECK_HIP(rocprim::exclusive_scan(temp_storage, temp_storage_size, input_data, output_data,
                                                 init_value, nb_item, op, stream));
      else
        ARCANE_CHECK_HIP(rocprim::inclusive_scan(temp_storage, temp_storage_size, input_data, output_data,
                                                 nb_item, op, stream));
    }
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      DataType sum = init_value;
      for (Int32 i = 0; i < nb_item; ++i) {
        DataType v = *input_data;
        if constexpr (IsExclusive) {
          *output_data = sum;
          sum = op(v, sum);
        }
        else {
          sum = op(v, sum);
          *output_data = sum;
        }
        ++input_data;
        ++output_data;
      }
    } break;
    default:
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
  }

 private:

  RunQueue m_queue;
  GenericDeviceStorage m_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithmes de scan exclusif ou inclusif sur accélérateurs.
 *
 * Voir https://en.wikipedia.org/wiki/Prefix_sum.
 *
 * Dans les méthodes suivantes, l'argument \a queue ne doit pas être nul.
 */
template <typename DataType>
class Scanner
{
 public:

  //! Somme exclusive
  static void exclusiveSum(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<true>(queue, input, output, impl::ScannerSumOperator<DataType>{});
  }
  //! Minimum exclusif
  static void exclusiveMin(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<true>(queue, input, output, impl::ScannerMinOperator<DataType>{});
  }
  //! Maximum exclusif
  static void exclusiveMax(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<true>(queue, input, output, impl::ScannerMaxOperator<DataType>{});
  }
  //! Somme inclusive
  static void inclusiveSum(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<false>(queue, input, output, impl::ScannerSumOperator<DataType>{});
  }
  //! Minimum inclusif
  static void inclusiveMin(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<false>(queue, input, output, impl::ScannerMinOperator<DataType>{});
  }
  //! Maximum inclusif
  static void inclusiveMax(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<false>(queue, input, output, impl::ScannerMaxOperator<DataType>{});
  }

 private:

  template <bool IsExclusive, typename Operator>
  static void _applyArray(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output, const Operator& op)
  {
    ARCANE_CHECK_POINTER(queue);
    impl::ScannerImpl scanner(*queue);
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());
    const DataType* input_data = input.data();
    DataType* output_data = output.data();
    DataType init_value = op.initialValue();
    scanner.apply<IsExclusive>(nb_item, input_data, output_data, init_value, op);
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
