﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
#include "arcane/accelerator/RunCommandLaunchInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
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
  static DataType defaultValue() { return {}; }
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
  static DataType defaultValue() { return std::numeric_limits<DataType>::max(); }
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
  static DataType defaultValue() { return std::numeric_limits<DataType>::lowest(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer un scan exlusif ou inclusif avec un opérateur spécifique.
 */
class ScannerImpl
{
 public:

  explicit ScannerImpl(const RunQueue& queue)
  : m_queue(queue)
  {}

 public:

  template <bool IsExclusive, typename InputIterator, typename OutputIterator,
            typename Operator, typename DataType>
  void apply(Int32 nb_item, InputIterator input_data, OutputIterator output_data,
             DataType init_value, Operator op, const TraceInfo& trace_info)
  {
    RunCommand command = makeCommand(m_queue);
    command << trace_info;
    impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
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
    launch_info.endExecute();
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
    _applyArray<true>(queue, input, output, ScannerSumOperator<DataType>{});
  }
  //! Minimum exclusif
  static void exclusiveMin(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<true>(queue, input, output, ScannerMinOperator<DataType>{});
  }
  //! Maximum exclusif
  static void exclusiveMax(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<true>(queue, input, output, ScannerMaxOperator<DataType>{});
  }
  //! Somme inclusive
  static void inclusiveSum(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<false>(queue, input, output, ScannerSumOperator<DataType>{});
  }
  //! Minimum inclusif
  static void inclusiveMin(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<false>(queue, input, output, ScannerMinOperator<DataType>{});
  }
  //! Maximum inclusif
  static void inclusiveMax(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output)
  {
    _applyArray<false>(queue, input, output, ScannerMaxOperator<DataType>{});
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
    DataType init_value = op.defaultValue();
    scanner.apply<IsExclusive>(nb_item, input_data, output_data, init_value, op, TraceInfo{});
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithmes de scan exclusif ou inclusif sur accélérateurs.
 *
 * Voir https://en.wikipedia.org/wiki/Prefix_sum.
 *
 * Dans les méthodes de scan, les valeurs entre les entrées et les sorties
 * ne doivent pas se chevaucher.
 */
class GenericScanner
{
 public:

  /*!
   * \brief Itérateur sur une lambda pour positionner une valeur via un index.
   */
  template <typename DataType, typename SetterLambda>
  class SetterLambdaIterator
  {
   public:

    //! Permet de positionner un élément de l'itérateur de sortie
    class Setter
    {
     public:

      ARCCORE_HOST_DEVICE explicit Setter(const SetterLambda& s, Int32 index)
      : m_index(index)
      , m_lambda(s)
      {}
      ARCCORE_HOST_DEVICE void operator=(const DataType& value)
      {
        m_lambda(m_index, value);
      }
      Int32 m_index = 0;
      SetterLambda m_lambda;
    };

    using value_type = Setter;
    using iterator_category = std::random_access_iterator_tag;
    using reference = Setter;
    using difference_type = ptrdiff_t;
    using ThatClass = SetterLambdaIterator<DataType, SetterLambda>;

   public:

    ARCCORE_HOST_DEVICE SetterLambdaIterator(const SetterLambda& s)
    : m_lambda(s)
    {}
    ARCCORE_HOST_DEVICE explicit SetterLambdaIterator(const SetterLambda& s, Int32 v)
    : m_index(v)
    , m_lambda(s)
    {}

   public:

    ARCCORE_HOST_DEVICE ThatClass& operator++()
    {
      ++m_index;
      return (*this);
    }
    ARCCORE_HOST_DEVICE ThatClass operator+(Int32 x)
    {
      return ThatClass(m_lambda, m_index + x);
    }
    ARCCORE_HOST_DEVICE ThatClass operator-(Int32 x)
    {
      return ThatClass(m_lambda, m_index - x);
    }
    ARCCORE_HOST_DEVICE Setter operator*() const
    {
      return Setter(m_lambda, m_index);
    }
    ARCCORE_HOST_DEVICE value_type operator[](Int32 x) const { return Setter(m_lambda, m_index + x); }

   private:

    Int32 m_index = 0;
    SetterLambda m_lambda;
  };

  /*!
   * \brief Itérateur sur une lambda pour récupérer une valeur via un index.
   */
  template <typename DataType, typename GetterLambda>
  class GetterLambdaIterator
  {
   public:

    using value_type = DataType;
    using iterator_category = std::random_access_iterator_tag;
    using reference = DataType&;
    using difference_type = ptrdiff_t;
    using ThatClass = GetterLambdaIterator<DataType, GetterLambda>;

   public:

    ARCCORE_HOST_DEVICE GetterLambdaIterator(const GetterLambda& s)
    : m_lambda(s)
    {}
    ARCCORE_HOST_DEVICE explicit GetterLambdaIterator(const GetterLambda& s, Int32 v)
    : m_index(v)
    , m_lambda(s)
    {}

   public:

    ARCCORE_HOST_DEVICE ThatClass& operator++()
    {
      ++m_index;
      return (*this);
    }
    ARCCORE_HOST_DEVICE ThatClass operator+(Int32 x)
    {
      return ThatClass(m_lambda, m_index + x);
    }
    ARCCORE_HOST_DEVICE ThatClass operator-(Int32 x)
    {
      return ThatClass(m_lambda, m_index - x);
    }
    ARCCORE_HOST_DEVICE value_type operator*() const
    {
      return m_lambda(m_index);
    }
    ARCCORE_HOST_DEVICE value_type operator[](Int32 x) const { return m_lambda(m_index + x); }

   private:

    Int32 m_index = 0;
    GetterLambda m_lambda;
  };

 public:

  explicit GenericScanner(const RunQueue& queue)
  : m_queue(queue)
  {}

 public:

  template <typename DataType, typename GetterLambda, typename SetterLambda, typename Operator>
  void applyWithIndexExclusive(Int32 nb_value, const DataType& initial_value,
                               const GetterLambda& getter_lambda,
                               const SetterLambda& setter_lambda,
                               const Operator& op_lambda,
                               const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex<true>(nb_value, initial_value, getter_lambda, setter_lambda, op_lambda, trace_info);
  }

  template <typename DataType, typename GetterLambda, typename SetterLambda, typename Operator>
  void applyWithIndexInclusive(Int32 nb_value, const DataType& initial_value,
                               const GetterLambda& getter_lambda,
                               const SetterLambda& setter_lambda,
                               const Operator& op_lambda,
                               const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex<false>(nb_value, initial_value, getter_lambda, setter_lambda, op_lambda, trace_info);
  }

  template <typename InputDataType, typename OutputDataType, typename Operator>
  void applyExclusive(const InputDataType& initial_value,
                      SmallSpan<const std::remove_cvref_t<InputDataType>> input,
                      SmallSpan<OutputDataType> output,
                      const Operator& op_lambda,
                      const TraceInfo& trace_info = TraceInfo())
  {
    _apply<true>(initial_value, input, output, op_lambda, trace_info);
  }

  template <typename InputDataType, typename OutputDataType, typename Operator>
  void applyInclusive(const InputDataType& initial_value,
                      SmallSpan<const std::remove_cvref_t<InputDataType>> input,
                      SmallSpan<OutputDataType> output,
                      const Operator& op_lambda,
                      const TraceInfo& trace_info = TraceInfo())
  {
    _apply<false>(initial_value, input, output, op_lambda, trace_info);
  }

 private:

  template <bool IsExclusive, typename DataType, typename GetterLambda, typename SetterLambda, typename Operator>
  void _applyWithIndex(Int32 nb_value, const DataType& initial_value,
                       const GetterLambda& getter_lambda,
                       const SetterLambda& setter_lambda,
                       const Operator& op_lambda,
                       const TraceInfo& trace_info)
  {
    GetterLambdaIterator<DataType, GetterLambda> input_iter(getter_lambda);
    SetterLambdaIterator<DataType, SetterLambda> output_iter(setter_lambda);
    impl::ScannerImpl scanner(m_queue);
    scanner.apply<IsExclusive>(nb_value, input_iter, output_iter, initial_value, op_lambda, trace_info);
  }

  template <bool IsExclusive, typename InputDataType, typename OutputDataType, typename Operator>
  void _apply(const InputDataType& initial_value,
              SmallSpan<const InputDataType> input,
              SmallSpan<OutputDataType> output,
              const Operator& op,
              const TraceInfo& trace_info = TraceInfo())
  {
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());
    auto* input_data = input.data();
    auto* output_data = output.data();
    impl::ScannerImpl scanner(m_queue);
    scanner.apply<IsExclusive>(nb_item, input_data, output_data, initial_value, op, trace_info);
  }

 private:

  RunQueue m_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
