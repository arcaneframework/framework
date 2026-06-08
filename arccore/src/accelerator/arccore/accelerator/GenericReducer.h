// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericReducer.h                                            (C) 2000-2026 */
/*                                                                           */
/* Reduction management for accelerators.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_GENERICREDUCER_H
#define ARCCORE_ACCELERATOR_GENERICREDUCER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FatalErrorException.h"

#include "arccore/common/NumArray.h"
#include "arccore/common/accelerator/RunQueue.h"

#include "arccore/accelerator/Reduce.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{
template <typename DataType>
class GenericReducerIf;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Class to determine the 'Reducer2' instance to use based on the operator.
// To be specialized.
template <typename DataType, typename Operator>
class ReduceOperatorToReducerTypeTraits;

template <typename DataType>
class ReduceOperatorToReducerTypeTraits<DataType, MaxOperator<DataType>>
{
 public:

  using ReducerType = ReducerMax2<DataType>;
};
template <typename DataType>
class ReduceOperatorToReducerTypeTraits<DataType, MinOperator<DataType>>
{
 public:

  using ReducerType = ReducerMin2<DataType>;
};
template <typename DataType>
class ReduceOperatorToReducerTypeTraits<DataType, SumOperator<DataType>>
{
 public:

  using ReducerType = ReducerSum2<DataType>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for performing a reduction.
 *
 * Contains the necessary arguments to perform a reduction.
 */
template <typename DataType>
class GenericReducerBase
{
  friend class GenericReducerIf<DataType>;

 public:

  GenericReducerBase(const RunQueue& queue)
  : m_queue(queue)
  {}

 protected:

  DataType _reducedValue() const
  {
    m_queue.barrier();
    return m_host_reduce_storage[0];
  }

  void _allocate()
  {
    eMemoryResource r = eMemoryResource::HostPinned;
    if (m_host_reduce_storage.memoryRessource() != r)
      m_host_reduce_storage = NumArray<DataType, MDDim1>(r);
    m_host_reduce_storage.resize(1);
  }

 protected:

  RunQueue m_queue;
  GenericDeviceStorage m_algo_storage;
  DeviceStorage<DataType> m_device_reduce_storage;
  NumArray<DataType, MDDim1> m_host_reduce_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class for partitioning a list.
 *
 * The list is partitioned into two lists.
 *
 * \a DataType is the data type.
 */
template <typename DataType>
class GenericReducerIf
{
  // TODO: Perform malloc on the queue's associated device.
  //       and also check if mallocAsync() can be used.

 public:

  template <typename InputIterator, typename ReduceOperator>
  void apply(GenericReducerBase<DataType>& s, Int32 nb_item, const DataType& init_value,
             InputIterator input_iter, ReduceOperator reduce_op, const TraceInfo& trace_info)
  {
    RunQueue& queue = s.m_queue;
    RunCommand command = makeCommand(queue);
    command << trace_info;
    Impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    eExecutionPolicy exec_policy = queue.executionPolicy();
    switch (exec_policy) {
#if defined(ARCCORE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = Impl::CudaUtils::toNativeStream(queue);
      DataType* reduced_value_ptr = nullptr;
      // First call to determine the size for allocation
      ARCCORE_CHECK_CUDA(::cub::DeviceReduce::Reduce(nullptr, temp_storage_size, input_iter, reduced_value_ptr,
                                                     nb_item, reduce_op, init_value, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      reduced_value_ptr = s.m_device_reduce_storage.allocate();
      ARCCORE_CHECK_CUDA(::cub::DeviceReduce::Reduce(s.m_algo_storage.address(), temp_storage_size,
                                                     input_iter, reduced_value_ptr, nb_item,
                                                     reduce_op, init_value, stream));
      s.m_device_reduce_storage.copyToAsync(s.m_host_reduce_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      hipStream_t stream = Impl::HipUtils::toNativeStream(queue);
      DataType* reduced_value_ptr = nullptr;
      // First call to determine the size for allocation
      ARCCORE_CHECK_HIP(rocprim::reduce(nullptr, temp_storage_size, input_iter, reduced_value_ptr, init_value,
                                        nb_item, reduce_op, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      reduced_value_ptr = s.m_device_reduce_storage.allocate();

      ARCCORE_CHECK_HIP(rocprim::reduce(s.m_algo_storage.address(), temp_storage_size, input_iter, reduced_value_ptr, init_value,
                                        nb_item, reduce_op, stream));
      s.m_device_reduce_storage.copyToAsync(s.m_host_reduce_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      {
        RunCommand command2 = makeCommand(queue);
        using ReducerType = typename ReduceOperatorToReducerTypeTraits<DataType, ReduceOperator>::ReducerType;
        ReducerType reducer(command2);
        command2 << RUNCOMMAND_LOOP1(iter, nb_item, reducer)
        {
          auto [i] = iter();
          reducer.combine(input_iter[i]);
        };
        queue.barrier();
        s.m_host_reduce_storage[0] = reducer.reducedValue();
      }
    } break;
#endif
    case eExecutionPolicy::Thread:
      // Not yet implemented in multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      DataType reduced_value = init_value;
      for (Int32 i = 0; i < nb_item; ++i) {
        reduced_value = reduce_op(reduced_value, *input_iter);
        ++input_iter;
      }
      s.m_host_reduce_storage[0] = reduced_value;
    } break;
    default:
      ARCCORE_FATAL(getBadPolicyMessage(exec_policy));
    }
    launch_info.endExecute();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic accelerator reduction algorithm.
 *
 * The reduction is performed via calls to applyMin(), applyMax(), applySum(),
 * applyMinWithIndex(), applyMaxWithIndex() or applySumWithIndex(). These
 * methods are asynchronous. After reduction, it is possible to retrieve the
 * reduced value via reducedValue(). The call to reducedValue() blocks until
 * the reduction is complete.
 *
 * Instances of this class can be used multiple times.
 *
 * Here is an example to calculate the sum of an array of 50 elements:
 *
 * \code
 * using namespace Arcane;
 * const Int32 nb_value(50);
 * Arcane::NumArray<Real, MDDim1> t1(nv_value);
 * Arcane::SmallSpan<const Real> t1_view(t1);
 * Arcane::RunQueue queue = ...;
 * Arcane::Accelerator::GenericReducer<Real> reducer(queue);
 *
 * // Direct calculation
 * reducer.applySum(t1_view);
 * std::cout << "Sum is '" << reducer.reducedValue() << "\n";
 *
 * // Calculation with lambda
 * auto getter_func = [=] ARCCORE_HOST_DEVICE(Int32 index) -> Real
 * {
 *   return t1_view[index];
 * }
 * reducer.applySumWithIndex(nb_value,getter_func);
 * std::cout << "Sum is '" << reducer.reducedValue() << "\n";
 * \endcode
 */
template <typename DataType>
class GenericReducer
: private Impl::GenericReducerBase<DataType>
{
 public:

  explicit GenericReducer(const RunQueue& queue)
  : Impl::GenericReducerBase<DataType>(queue)
  {
    this->_allocate();
  }

 public:

  //! Applies a 'Min' reduction on the values \a values
  void applyMin(SmallSpan<const DataType> values, const TraceInfo& trace_info = TraceInfo())
  {
    _apply(values.size(), values.data(), Impl::MinOperator<DataType>{}, trace_info);
  }

  //! Applies a 'Max' reduction on the values \a values
  void applyMax(SmallSpan<const DataType> values, const TraceInfo& trace_info = TraceInfo())
  {
    _apply(values.size(), values.data(), Impl::MaxOperator<DataType>{}, trace_info);
  }

  //! Applies a 'Sum' reduction on the values \a values
  void applySum(SmallSpan<const DataType> values, const TraceInfo& trace_info = TraceInfo())
  {
    _apply(values.size(), values.data(), Impl::SumOperator<DataType>{}, trace_info);
  }

  //! Applies a 'Min' reduction on the values selected by \a select_lambda
  template <typename SelectLambda>
  void applyMinWithIndex(Int32 nb_value, const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex(nb_value, select_lambda, Impl::MinOperator<DataType>{}, trace_info);
  }

  //! Applies a 'Max' reduction on the values selected by \a select_lambda
  template <typename SelectLambda>
  void applyMaxWithIndex(Int32 nb_value, const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex(nb_value, select_lambda, Impl::MaxOperator<DataType>{}, trace_info);
  }

  //! Applies a 'Sum' reduction on the values selected by \a select_lambda
  template <typename SelectLambda>
  void applySumWithIndex(Int32 nb_value, const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex(nb_value, select_lambda, Impl::SumOperator<DataType>{}, trace_info);
  }

  //! Reduction value
  DataType reducedValue()
  {
    m_is_already_called = false;
    return this->_reducedValue();
  }

 private:

  bool m_is_already_called = false;

 private:

  template <typename InputIterator, typename ReduceOperator>
  void _apply(Int32 nb_value, InputIterator input_iter, ReduceOperator reduce_op, const TraceInfo& trace_info)
  {
    _setCalled();
    Impl::GenericReducerBase<DataType>* base_ptr = this;
    Impl::GenericReducerIf<DataType> gf;
    DataType init_value = reduce_op.defaultValue();
    gf.apply(*base_ptr, nb_value, init_value, input_iter, reduce_op, trace_info);
  }

  template <typename GetterLambda, typename ReduceOperator>
  void _applyWithIndex(Int32 nb_value, const GetterLambda& getter_lambda,
                       ReduceOperator reduce_op, const TraceInfo& trace_info)
  {
    _setCalled();
    Impl::GenericReducerBase<DataType>* base_ptr = this;
    Impl::GenericReducerIf<DataType> gf;
    Impl::GetterLambdaIterator<DataType, GetterLambda> input_iter(getter_lambda);
    DataType init_value = reduce_op.defaultValue();
    gf.apply(*base_ptr, nb_value, init_value, input_iter, reduce_op, trace_info);
  }

  void _setCalled()
  {
    if (m_is_already_called)
      ARCCORE_FATAL("apply() has already been called for this instance");
    m_is_already_called = true;
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
