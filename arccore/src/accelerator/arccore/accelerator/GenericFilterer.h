// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericFilterer.h                                           (C) 2000-2026 */
/*                                                                           */
/* Filtering algorithm.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_GENERICFILTERER_H
#define ARCCORE_ACCELERATOR_GENERICFILTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/ScanImpl.h"
#include "arccore/accelerator/MultiThreadAlgo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{
//#define ARCCORE_USE_SCAN_ONEDPL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for performing filtering.
 *
 * Contains the necessary arguments to perform the filtering.
 */
class ARCCORE_ACCELERATOR_EXPORT GenericFilteringBase
{
  template <typename DataType, typename FlagType, typename OutputDataType>
  friend class GenericFilteringFlag;
  friend class GenericFilteringIf;
  friend class SyclGenericFilteringImpl;

 public:
 protected:

  GenericFilteringBase();

 protected:

  Int32 _nbOutputElement();
  void _allocate();
  void _allocateTemporaryStorage(size_t size);
  int* _getDeviceNbOutPointer();
  void _copyDeviceNbOutToHostNbOut();
  void _setCalled();
  bool _checkEmpty(Int32 nb_value);

 protected:

  //! Execution queue. Must not be null.
  RunQueue m_queue;
  // Working memory for the filtering algorithm.
  GenericDeviceStorage m_algo_storage;
  //! Device memory for the number of filtered values
  DeviceStorage<int> m_device_nb_out_storage;
  //! Host memory for the number of filtered values
  NumArray<Int32, MDDim1> m_host_nb_out_storage;
  /*!
   * \brief Indicates which memory is used for the number of filtered values.
   *
   * If true, it uses \a m_host_nb_out_storage directly. Otherwise, it uses
   * m_device_nb_out_storage and performs an asynchronous copy after filtering to
   * copy the value into m_host_nb_out_storage.
   */
  bool m_use_direct_host_storage = true;

  //! Indicates if a call is in progress
  bool m_is_already_called = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)
//! Implementation for SYCL
class SyclGenericFilteringImpl
{
 public:

  template <typename SelectLambda, typename InputIterator, typename OutputIterator>
  static void apply(GenericFilteringBase& s, Int32 nb_item, InputIterator input_iter,
                    OutputIterator output_iter, SelectLambda select_lambda)
  {
    RunQueue queue = s.m_queue;
    using DataType = std::iterator_traits<OutputIterator>::value_type;
#if defined(ARCCORE_USE_SCAN_ONEDPL) && defined(ARCCORE_HAS_ONEDPL)
    sycl::queue true_queue = AcceleratorUtils::toSyclNativeStream(queue);
    auto policy = oneapi::dpl::execution::make_device_policy(true_queue);
    auto out_iter = oneapi::dpl::copy_if(policy, input_iter, input_iter + nb_item, output_iter, select_lambda);
    Int32 nb_output = static_cast<Int32>(out_iter - output_iter);
    s.m_host_nb_out_storage[0] = nb_output;
#else
    NumArray<Int32, MDDim1> scan_input_data(nb_item);
    NumArray<Int32, MDDim1> scan_output_data(nb_item);
    SmallSpan<Int32> in_scan_data = scan_input_data.to1DSmallSpan();
    SmallSpan<Int32> out_scan_data = scan_output_data.to1DSmallSpan();
    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, nb_item)
      {
        auto [i] = iter();
        in_scan_data[i] = select_lambda(input_iter[i]) ? 1 : 0;
      };
    }
    queue.barrier();
    SyclScanner<false /*is_exclusive*/, Int32, ScannerSumOperator<Int32>> scanner;
    scanner.doScan(queue, in_scan_data, out_scan_data, 0);
    // The value of 'out_data' for the last element (nb_item-1) contains the filter size
    Int32 nb_output = out_scan_data[nb_item - 1];
    s.m_host_nb_out_storage[0] = nb_output;

    const bool do_verbose = false;
    if (do_verbose && nb_item < 1500)
      for (int i = 0; i < nb_item; ++i) {
        std::cout << "out_data i=" << i << " out_data=" << out_scan_data[i]
                  << " in_data=" << in_scan_data[i] << " value=" << input_iter[i] << "\n ";
      }
    // Copy indices corresponding to the filter from 'out_data' to 'in_data'.
    // Since 'output_iter' and 'input_iter' may overlap, it
    // is necessary to make an intermediate copy.
    // TODO: detect this and only perform the copy if necessary.
    NumArray<DataType, MDDim1> out_copy(eMemoryResource::Device);
    out_copy.resize(nb_output);
    auto out_copy_view = out_copy.to1DSpan();
    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, nb_item)
      {
        auto [i] = iter();
        if (in_scan_data[i] == 1)
          out_copy_view[out_scan_data[i] - 1] = input_iter[i];
      };
    }
    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, nb_output)
      {
        auto [i] = iter();
        output_iter[i] = out_copy_view[i];
      };
    }
    // Necessary because of 'out_copy'. This can be removed with an
    // temporary allocation.
    queue.barrier();
#endif
  }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class for performing filtering
 *
 * \a DataType is the data type.
 * \a FlagType is the type of the filter array.
 */
template <typename DataType, typename FlagType, typename OutputDataType>
class GenericFilteringFlag
{
 public:

  void apply(GenericFilteringBase& s, SmallSpan<const DataType> input,
             SmallSpan<OutputDataType> output, SmallSpan<const FlagType> flag)
  {
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCCORE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());
    [[maybe_unused]] const DataType* input_data = input.data();
    [[maybe_unused]] DataType* output_data = output.data();
    [[maybe_unused]] const FlagType* flag_data = flag.data();
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    RunQueue queue = s.m_queue;
    exec_policy = queue.executionPolicy();
    switch (exec_policy) {
#if defined(ARCCORE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = AcceleratorUtils::toCudaNativeStream(queue);
      // First call to determine the size for allocation
      int* nb_out_ptr = nullptr;
      ARCCORE_CHECK_CUDA(::cub::DeviceSelect::Flagged(nullptr, temp_storage_size,
                                                      input_data, flag_data, output_data, nb_out_ptr, nb_item, stream));

      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();
      ARCCORE_CHECK_CUDA(::cub::DeviceSelect::Flagged(s.m_algo_storage.address(), temp_storage_size,
                                                      input_data, flag_data, output_data, nb_out_ptr, nb_item, stream));
      s._copyDeviceNbOutToHostNbOut();
    } break;
#endif
#if defined(ARCCORE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // First call to determine the size for allocation
      hipStream_t stream = AcceleratorUtils::toHipNativeStream(queue);
      int* nb_out_ptr = nullptr;
      ARCCORE_CHECK_HIP(rocprim::select(nullptr, temp_storage_size, input_data, flag_data, output_data,
                                        nb_out_ptr, nb_item, stream));

      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();

      ARCCORE_CHECK_HIP(rocprim::select(s.m_algo_storage.address(), temp_storage_size, input_data, flag_data, output_data,
                                        nb_out_ptr, nb_item, stream));
      s._copyDeviceNbOutToHostNbOut();
    } break;
#endif
#if defined(ARCCORE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      Impl::IndexIterator iter2(0);
      auto filter_lambda = [=](Int32 input_index) -> bool { return flag[input_index] != 0; };
      auto setter_lambda = [=](Int32 input_index, Int32 output_index) { output[output_index] = input[input_index]; };
      Impl::SetterLambdaIterator<decltype(setter_lambda)> out(setter_lambda);
      SyclGenericFilteringImpl::apply(s, nb_item, iter2, out, filter_lambda);
    } break;
#endif
    case eExecutionPolicy::Thread:
      // Not yet implemented in multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      Int32 index = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        if (flag[i] != 0) {
          output[index] = input[i];
          ++index;
        }
      }
      s.m_host_nb_out_storage[0] = index;
    } break;
    default:
      ARCCORE_FATAL(getBadPolicyMessage(exec_policy));
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class for performing filtering
 *
 * \a DataType is the data type.
 * \a FlagType is the type of the filter array.
 */
class GenericFilteringIf
{
 public:

  /*!
   * \brief Applies the filter.
   *
   * If \a InPlace is true, then OutputIterator equals InputIterator and we
   * update \a input_iter directly.
   */
  template <bool InPlace, typename SelectLambda, typename InputIterator, typename OutputIterator>
  void apply(GenericFilteringBase& s, Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
             const SelectLambda& select_lambda, const TraceInfo& trace_info)
  {
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    RunQueue queue = s.m_queue;
    exec_policy = queue.executionPolicy();
    RunCommand command = makeCommand(queue);
    command << trace_info;
    Impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    switch (exec_policy) {
#if defined(ARCCORE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = Impl::CudaUtils::toNativeStream(queue);
      // First call to determine the size for allocation
      int* nb_out_ptr = nullptr;
      if constexpr (InPlace)
        ARCCORE_CHECK_CUDA(::cub::DeviceSelect::If(nullptr, temp_storage_size,
                                                   input_iter, nb_out_ptr, nb_item,
                                                   select_lambda, stream));
      else
        ARCCORE_CHECK_CUDA(::cub::DeviceSelect::If(nullptr, temp_storage_size,
                                                   input_iter, output_iter, nb_out_ptr, nb_item,
                                                   select_lambda, stream));

      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();
      if constexpr (InPlace)
        ARCCORE_CHECK_CUDA(::cub::DeviceSelect::If(s.m_algo_storage.address(), temp_storage_size,
                                                   input_iter, nb_out_ptr, nb_item,
                                                   select_lambda, stream));
      else
        ARCCORE_CHECK_CUDA(::cub::DeviceSelect::If(s.m_algo_storage.address(), temp_storage_size,
                                                   input_iter, output_iter, nb_out_ptr, nb_item,
                                                   select_lambda, stream));

      s._copyDeviceNbOutToHostNbOut();
    } break;
#endif
#if defined(ARCCORE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // First call to determine the size for allocation
      hipStream_t stream = Impl::HipUtils::toNativeStream(queue);
      int* nb_out_ptr = nullptr;
      // NOTE: there is no specific in-place 'select' version.
      // It is possible that \a input_iter and \a output_iter
      // have the same value.
      ARCCORE_CHECK_HIP(rocprim::select(nullptr, temp_storage_size, input_iter, output_iter,
                                        nb_out_ptr, nb_item, select_lambda, stream));
      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();
      ARCCORE_CHECK_HIP(rocprim::select(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                        nb_out_ptr, nb_item, select_lambda, 0));
      s._copyDeviceNbOutToHostNbOut();
    } break;
#endif
#if defined(ARCCORE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      SyclGenericFilteringImpl::apply(s, nb_item, input_iter, output_iter, select_lambda);
    } break;
#endif
    case eExecutionPolicy::Thread:
      if (nb_item > 500) {
        MultiThreadAlgo scanner;
        Int32 v = scanner.doFilter<InPlace>(launch_info.loopRunInfo(), nb_item, input_iter, output_iter, select_lambda);
        s.m_host_nb_out_storage[0] = v;
        break;
      }
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      Int32 index = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        if (select_lambda(*input_iter)) {
          *output_iter = *input_iter;
          ++index;
          ++output_iter;
        }
        ++input_iter;
      }
      s.m_host_nb_out_storage[0] = index;
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

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic filtering algorithm on accelerator.
 */
class GenericFilterer
: private Impl::GenericFilteringBase
{

 public:

  /*!
   * \brief Creates an instance.
   *
   * \pre queue!=nullptr
   */
  ARCCORE_DEPRECATED_REASON("Y2024: Use GenericFilterer(const RunQueue&) instead")
  explicit GenericFilterer(RunQueue* queue)
  {
    ARCCORE_CHECK_POINTER(queue);
    m_queue = *queue;
    _allocate();
  }

  /*!
   * \brief Creates an instance.
   *
   * \pre queue!=nullptr
   */
  explicit GenericFilterer(const RunQueue& queue)
  {
    m_queue = queue;
    _allocate();
  }

 public:

  /*!
   * \brief Applies a filter.
   *
   * Filters all elements in \a input for which \a flag is not 0 and
   * fills \a output with the filtered values. \a output must be large enough
   * to hold all filtered elements.
   *
   * The sequential algorithm is as follows:
   *
   * \code
   * Int32 index = 0;
   * for (Int32 i = 0; i < nb_item; ++i) {
   *   if (flag[i] != 0) {
   *     output[index] = input[i];
   *     ++index;
   *   }
   * }
   * return index;
   * \endcode
   *
   * You must call the nbOutputElement() method to get the number of elements
   * after filtering.
   */
  template <typename InputDataType, typename OutputDataType, typename FlagType>
  void apply(SmallSpan<const InputDataType> input, SmallSpan<OutputDataType> output, SmallSpan<const FlagType> flag)
  {
    const Int32 nb_value = input.size();
    if (output.size() != nb_value)
      ARCCORE_FATAL("Sizes are not equals: input={0} output={1}", nb_value, output.size());
    if (flag.size() != nb_value)
      ARCCORE_FATAL("Sizes are not equals: input={0} flag={1}", nb_value, flag.size());

    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericFilteringBase* base_ptr = this;
    Impl::GenericFilteringFlag<InputDataType, FlagType, OutputDataType> gf;
    gf.apply(*base_ptr, input, output, flag);
  }

  /*!
   * \brief Applies a filter.
   *
   * Filters all elements in \a input for which \a select_lambda equals \a true and
   * fills \a output with the filtered values. \a output must be large enough
   * to hold all filtered elements. The memory regions associated with \a input and
   * \a output must not overlap.
   *
   * \a select_lambda must have an operator `ARCCORE_HOST_DEVICE bool operator()(const DataType& v) const'.
   *
   * For example, the following lambda keeps only elements whose
   * value is greater than 569.
   *
   * \code
   * auto filter_lambda = [] ARCCORE_HOST_DEVICE (const DataType& x) -> bool {
   *   return (x > 569.0);
   * };
   * \endcode
   *
   * The sequential algorithm is as follows:
   *
   * \code
   * Int32 index = 0;
   * for (Int32 i = 0; i < nb_item; ++i) {
   *   if (select_lambda(input[i])) {
   *     output[index] = input[i];
   *     ++index;
   *   }
   * }
   * return index;
   * \endcode
   *
   * You must call the nbOutputElement() method to get the number of elements
   * after filtering.
   */
  template <typename DataType, typename SelectLambda>
  void applyIf(SmallSpan<const DataType> input, SmallSpan<DataType> output,
               const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    const Int32 nb_value = input.size();
    if (output.size() != nb_value)
      ARCCORE_FATAL("Sizes are not equals: input={0} output={1}", nb_value, output.size());
    if (input.data() == output.data())
      ARCCORE_FATAL("Input and Output are the same. Use in place overload instead");
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericFilteringBase* base_ptr = this;
    Impl::GenericFilteringIf gf;
    gf.apply<false>(*base_ptr, nb_value, input.data(), output.data(), select_lambda, trace_info);
  }

  /*!
   * \brief Applies an in-place filter.
   *
   * This method is identical to applyIf(SmallSpan<const DataType>, SmallSpan<DataType>,
   * const SelectLambda&, const TraceInfo& trace_info) but the filtered values are
   * directly copied into the \a input_output array.
   */
  template <typename DataType, typename SelectLambda>
  void applyIf(SmallSpan<DataType> input_output, const SelectLambda& select_lambda,
               const TraceInfo& trace_info = TraceInfo())
  {
    const Int32 nb_value = input_output.size();
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericFilteringBase* base_ptr = this;
    Impl::GenericFilteringIf gf;
    gf.apply<true>(*base_ptr, nb_value, input_output.data(), input_output.data(), select_lambda, trace_info);
  }

  /*!
   * \brief Applies a filter.
   *
   * This method is identical to Filterer::applyIf(SmallSpan<const DataType> input,
   * SmallSpan<DataType> output, const SelectLambda& select_lambda) but allows specifying an
   * \a input_iter iterator for the input and \a output_iter for the output.
   * The number of input entities is given by \a nb_value.
   *
   * The memory regions associated with \a input_iter and \a output_iter must not overlap.
   */
  template <typename InputIterator, typename OutputIterator, typename SelectLambda>
  void applyIf(Int32 nb_value, InputIterator input_iter, OutputIterator output_iter,
               const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericFilteringBase* base_ptr = this;
    Impl::GenericFilteringIf gf;
    gf.apply<false>(*base_ptr, nb_value, input_iter, output_iter, select_lambda, trace_info);
  }

  /*!
   * \brief Applies a filter with selection based on an index.
   *
   * This method allows filtering by specifying a lambda function for both
   * selection and assignment. The prototype for these lambdas is:
   *
   * \code
   * auto select_lambda = [=] ARCCORE_HOST_DEVICE (Int32 index) -> bool;
   * auto setter_lambda = [=] ARCCORE_HOST_DEVICE (Int32 input_index,Int32 output_index) -> void;
   * \endcode
   *
   * For example, if you want to copy the \a input array into \a output whose
   * values are greater than 25.0.
   *
   * \code
   * SmallSpan<Real> input = ...;
   * SmallSpan<Real> output = ...;
   * auto select_lambda = [=] ARCCORE_HOST_DEVICE (Int32 index) -> bool
   * {
   *   return input[index] > 25.0;
   * };
   * auto setter_lambda = [=] ARCCORE_HOST_DEVICE (Int32 input_index,Int32 output_index)
   * {
   *   output[output_index] = input[input_index];
   * };
   * Arcane::Accelerator::RunQueue* queue = ...;
   * Arcane::Accelerator::GenericFilterer filterer(queue);
   * filterer.applyWithIndex(input.size(), select_lambda, setter_lambda);
   * Int32 nb_out = filterer.nbOutputElement();
   * \endcode
   *
   * The memory regions associated with the input and output values must not overlap.
   */
  template <typename SelectLambda, typename SetterLambda>
  void applyWithIndex(Int32 nb_value, const SelectLambda& select_lambda,
                      const SetterLambda& setter_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericFilteringBase* base_ptr = this;
    Impl::GenericFilteringIf gf;
    Impl::IndexIterator input_iter;
    Impl::SetterLambdaIterator<SetterLambda> out(setter_lambda);
    gf.apply<false>(*base_ptr, nb_value, input_iter, out, select_lambda, trace_info);
  }

  /*!
   * \brief Number of output elements.
   *
   * \brief This method performs a barrier before retrieving the value.
   */
  Int32 nbOutputElement()
  {
    return _nbOutputElement();
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
