// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericPartitioner.h                                        (C) 2000-2026 */
/*                                                                           */
/* List partitioning algorithm.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_GENERICPARTITIONER_H
#define ARCCORE_ACCELERATOR_GENERICPARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotImplementedException.h"

#include "arccore/common/NumArray.h"
#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/accelerator/RunCommandLaunchInfo.h"

#include "arccore/accelerator/CommonUtils.h"
#if defined(ARCCORE_COMPILING_SYCL)
#include "arccore/accelerator/RunCommandLoop.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for performing filtering.
 *
 * Contains the necessary arguments to perform filtering.
 */
class ARCCORE_ACCELERATOR_EXPORT GenericPartitionerBase
{
  friend class GenericPartitionerIf;

 public:

  explicit GenericPartitionerBase(const RunQueue& queue);

 protected:

  Int32 _nbFirstPart() const;
  SmallSpan<const Int32> _nbParts() const;
  void _allocate();
  void _resetNbPart();

 protected:

  RunQueue m_queue;
  GenericDeviceStorage m_algo_storage;
  DeviceStorage<int, 2> m_device_nb_list1_storage;
  NumArray<Int32, MDDim1> m_host_nb_list1_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class for partitioning a list.
 */
class GenericPartitionerIf
{
 public:

  /*!
   * \brief Performs the partitioning of a list into two parts.
   */
  template <typename SelectLambda, typename InputIterator, typename OutputIterator>
  void apply(GenericPartitionerBase& s, Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
             const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    RunQueue queue = s.m_queue;
    eExecutionPolicy exec_policy = queue.executionPolicy();
    RunCommand command = makeCommand(queue);
    command << trace_info;
    Impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    switch (exec_policy) {
#if defined(ARCCORE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = Impl::CudaUtils::toNativeStream(&queue);
      // First call to determine the size for allocation
      int* nb_list1_ptr = nullptr;
      ARCCORE_CHECK_CUDA(::cub::DevicePartition::If(nullptr, temp_storage_size,
                                                    input_iter, output_iter, nb_list1_ptr, nb_item,
                                                    select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();
      ARCCORE_CHECK_CUDA(::cub::DevicePartition::If(s.m_algo_storage.address(), temp_storage_size,
                                                    input_iter, output_iter, nb_list1_ptr, nb_item,
                                                    select_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // First call to determine the size for allocation
      hipStream_t stream = Impl::HipUtils::toNativeStream(&queue);
      int* nb_list1_ptr = nullptr;
      ARCCORE_CHECK_HIP(rocprim::partition(nullptr, temp_storage_size, input_iter, output_iter,
                                           nb_list1_ptr, nb_item, select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();

      ARCCORE_CHECK_HIP(rocprim::partition(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                           nb_list1_ptr, nb_item, select_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
#if defined(ARCCORE_HAS_ONEDPL)
      // Only implemented for DPC++.
      // Currently (dpc++ 2025.0), there is no SYCL equivalent to
      // the cub or rocprim partitioning method.
      // Uses the 'stable_partition' function. However, this function
      // does not support memory solely on the accelerator. Furthermore,
      // InputIterator and OutputIterator must satisfy the
      // std::random_access_iterator concept, which is not the case (notably because there are
      // no copy operators or default constructors due to the lambda).
      // To avoid all these problems, temporary arrays are allocated
      // for the 'stable_sort' call and the values are copied to the output.
      using InputDataType = typename InputIterator::value_type;
      using DataType = typename OutputIterator::value_type;
      NumArray<DataType, MDDim1> tmp_output_numarray(nb_item);
      NumArray<bool, MDDim1> tmp_select_numarray(nb_item);
      auto tmp_output = tmp_output_numarray.to1DSmallSpan();
      auto tmp_select = tmp_select_numarray.to1DSmallSpan();
      {
        auto command = makeCommand(queue);
        command << RUNCOMMAND_LOOP1(iter, nb_item)
        {
          auto [i] = iter();
          tmp_output[i] = input_iter[i];
        };
      }
      auto tmp_select_lambda = [=](Int32 i) { return tmp_select[i]; };
      sycl::queue sycl_queue = Impl::SyclUtils::toNativeStream(queue);
      auto policy = oneapi::dpl::execution::make_device_policy(sycl_queue);
      auto output_after = oneapi::dpl::stable_partition(policy, tmp_output.begin(), tmp_output.end(), select_lambda);
      queue.barrier();
      Int32 nb_list1 = (output_after - tmp_output.begin());
      Int32 nb_list2 = nb_item - nb_list1;
      s.m_host_nb_list1_storage[0] = nb_list1;
      //std::cerr << "NbList1=" << nb_list1 << " NbList2=" << nb_list2 << "\n";
      {
        // Copy the filtered values into the output.
        // To be consistent with 'cub' and 'rocprim', the order of
        // the list values must be reversed for elements that do not meet the condition.
        // For this, a loop of size (nb_list1 + nb_list/2) is performed, and each
        // iteration for i>=nb_list1 handles two elements.
        auto command = makeCommand(queue);
        Int32 nb_iter2 = (nb_list2 / 2) + (nb_list2 % 2);
        //std::cout << "NB_ITER2=" << nb_iter2 << "\n";
        command << RUNCOMMAND_LOOP1(iter, (nb_list1 + nb_iter2))
        {
          auto [i] = iter();
          if (i >= nb_list1) {
            // Part of the list for values that do not meet the criterion.
            Int32 j = i - nb_list1;
            Int32 reverse_i = (nb_item - (j + 1));
            auto x1 = tmp_output[i];
            auto x2 = tmp_output[reverse_i];
            output_iter[i] = tmp_output[reverse_i];
            output_iter[reverse_i] = tmp_output[i];
          }
          else
            output_iter[i] = tmp_output[i];
        };
      }
      queue.barrier();
#else // ARCCORE_HAS_ONEDPL
      ARCCORE_THROW(NotImplementedException, "Partition is only implemented for SYCL back-end using oneDPL");
#endif // ARCCORE_HAS_ONEDPL
    } break;
#endif // ARCCORE_COMPILING_SYCL
    case eExecutionPolicy::Thread:
      // Not yet implemented in multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      auto saved_output_iter = output_iter;
      auto output2_iter = output_iter + nb_item;
      for (Int32 i = 0; i < nb_item; ++i) {
        auto v = *input_iter;
        if (select_lambda(v)) {
          *output_iter = v;
          ++output_iter;
        }
        else {
          --output2_iter;
          *output2_iter = v;
        }
        ++input_iter;
      }
      Int32 nb_list1 = static_cast<Int32>(output_iter - saved_output_iter);
      s.m_host_nb_list1_storage[0] = nb_list1;
    } break;
    default:
      ARCCORE_FATAL(getBadPolicyMessage(exec_policy));
    }
    launch_info.endExecute();
  }

  /*!
   * \brief Performs the partitioning of a list into three parts.
   */
  template <typename Select1Lambda, typename Select2Lambda,
            typename InputIterator, typename FirstOutputIterator,
            typename SecondOutputIterator, typename UnselectedIterator>
  void apply3(GenericPartitionerBase& s, Int32 nb_item,
              InputIterator input_iter,
              FirstOutputIterator first_output_iter,
              SecondOutputIterator second_output_iter,
              UnselectedIterator unselected_iter,
              const Select1Lambda& select1_lambda,
              const Select2Lambda& select2_lambda,
              const TraceInfo& trace_info = TraceInfo())
  {
    RunQueue queue = s.m_queue;
    eExecutionPolicy exec_policy = queue.executionPolicy();
    RunCommand command = makeCommand(queue);
    command << trace_info;
    Impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    switch (exec_policy) {
#if defined(ARCCORE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = Impl::CudaUtils::toNativeStream(&queue);
      // First call to determine the size for allocation
      int* nb_list1_ptr = nullptr;
      ARCCORE_CHECK_CUDA(::cub::DevicePartition::If(nullptr, temp_storage_size,
                                                    input_iter, first_output_iter, second_output_iter,
                                                    unselected_iter, nb_list1_ptr, nb_item,
                                                    select1_lambda, select2_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();
      ARCCORE_CHECK_CUDA(::cub::DevicePartition::If(s.m_algo_storage.address(), temp_storage_size,
                                                    input_iter, first_output_iter, second_output_iter,
                                                    unselected_iter, nb_list1_ptr, nb_item,
                                                    select1_lambda, select2_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // First call to determine the size for allocation
      hipStream_t stream = Impl::HipUtils::toNativeStream(&queue);
      int* nb_list1_ptr = nullptr;
      using namespace rocprim;
      ARCCORE_CHECK_HIP(::rocprim::partition_three_way(nullptr, temp_storage_size, input_iter, first_output_iter,
                                                       second_output_iter, unselected_iter,
                                                       nb_list1_ptr, nb_item, select1_lambda, select2_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();

      ARCCORE_CHECK_HIP(partition_three_way(s.m_algo_storage.address(), temp_storage_size, input_iter, first_output_iter,
                                            second_output_iter, unselected_iter, nb_list1_ptr, nb_item,
                                            select1_lambda, select2_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      ARCCORE_THROW(NotImplementedException, "3-way partition is not implemented for SYCL back-end");
    } break;
#endif
    case eExecutionPolicy::Thread:
      // Not yet implemented in multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      Int32 nb_first = 0;
      Int32 nb_second = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        auto v = *input_iter;
        bool is_1 = select1_lambda(v);
        bool is_2 = select2_lambda(v);
        if (is_1) {
          *first_output_iter = v;
          ++first_output_iter;
          ++nb_first;
        }
        else {
          if (is_2) {
            *second_output_iter = v;
            ++second_output_iter;
            ++nb_second;
          }
          else {
            *unselected_iter = v;
            ++unselected_iter;
          }
        }
        // Increment the iterator at the end because it is used for positioning
        ++input_iter;
      }
      s.m_host_nb_list1_storage[0] = nb_first;
      s.m_host_nb_list1_storage[1] = nb_second;
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
 * \brief Generic algorithm for partitioning a list.
 *
 * This class provides algorithms to partition a list into two or three parts
 * based on a criterion set by the user.
 */
class GenericPartitioner
: private Impl::GenericPartitionerBase
{
 public:

  explicit GenericPartitioner(const RunQueue& queue)
  : Impl::GenericPartitionerBase(queue)
  {
    _allocate();
  }

 public:

  /*!
   * \brief Performs a two-way partition of a list.
   *
   * The number of values in the list is given by \a nb_value.
   * The two lambda functions \a select_lambda and \a setter_lambda allow
   * partitioning and positioning the list values.
   *
   * After execution, it is possible to retrieve the number of elements
   * in the first part of the list via the \a nbFirstPart() method.
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerTwoWayIndex
   */
  template <typename SelectLambda, typename SetterLambda>
  void applyWithIndex(Int32 nb_value, const SetterLambda& setter_lambda,
                      const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericPartitionerBase* base_ptr = this;
    Impl::GenericPartitionerIf gf;

    Impl::IndexIterator input_iter;
    Impl::SetterLambdaIterator<SetterLambda> out(setter_lambda);
    gf.apply(*base_ptr, nb_value, input_iter, out, select_lambda, trace_info);
  }

  /*!
   * \brief Performs a two-way partition of a list.
   *
   * The number of values in the list is given by \a nb_value.
   * Input values are provided by the \a input_iter iterator and
   * output values by the \a output_iterator iterator. The \a select_lambda
   * function allows selecting the partition used:
   * if the return is \a true, the value will be in the first part of the list,
   * otherwise it will be in the second. The values in the second
   * part are sorted in reverse order of the input list.
   *
   * After execution, it is possible to retrieve the number of elements
   * in the first part of the list via the \a nbFirstPart() method.
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerTwoWayIf
   */
  template <typename InputIterator, typename OutputIterator, typename SelectLambda>
  void applyIf(Int32 nb_value, InputIterator input_iter, OutputIterator output_iter,
               const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericPartitionerBase* base_ptr = this;
    Impl::GenericPartitionerIf gf;
    gf.apply(*base_ptr, nb_value, input_iter, output_iter, select_lambda, trace_info);
  }

  /*!
   * \brief Performs a three-way partition of a list.
   *
   * The number of values in the list is given by \a nb_value.
   * The two lambda functions \a select1_lambda and \a select2_lambda allow
   * partitioning the list using the following algorithm:
   * - if select1_lambda() is true, the value will be positioned via \a setter1_lambda,
   * - otherwise if select2_lambda() is true, the value will be positioned via \a setter2_lambda,
   * - otherwise the value will be positioned via \a unselected_setter_lambda.
   *
   * The output lists are in the same order as the input.
   *
   * After execution, it is possible to retrieve the number of elements
   * in the first and second lists using the \a nbParts() method.
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerThreeWayIndex
   */
  template <typename Setter1Lambda, typename Setter2Lambda, typename UnselectedSetterLambda,
            typename Select1Lambda, typename Select2Lambda>
  void applyWithIndex(Int32 nb_value,
                      const Setter1Lambda setter1_lambda,
                      const Setter2Lambda setter2_lambda,
                      const UnselectedSetterLambda& unselected_setter_lambda,
                      const Select1Lambda& select1_lambda,
                      const Select2Lambda& select2_lambda,
                      const TraceInfo& trace_info = TraceInfo())
  {
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericPartitionerBase* base_ptr = this;
    Impl::GenericPartitionerIf gf;
    Impl::IndexIterator input_iter;
    Impl::SetterLambdaIterator<Setter1Lambda> setter1_wrapper(setter1_lambda);
    Impl::SetterLambdaIterator<Setter2Lambda> setter2_wrapper(setter2_lambda);
    Impl::SetterLambdaIterator<UnselectedSetterLambda> unselected_setter_wrapper(unselected_setter_lambda);
    gf.apply3(*base_ptr, nb_value, input_iter, setter1_wrapper, setter2_wrapper,
              unselected_setter_wrapper, select1_lambda, select2_lambda, trace_info);
  }

  /*!
   * \brief Performs a three-way partition of a list.
   *
   * The number of values in the list is given by \a nb_value.
   * The two lambda functions \a select1_lambda and \a select2_lambda allow
   * partitioning the list using the following algorithm:
   * - if select1_lambda() is true, the value is added to \a first_output_iter,
   * - otherwise if select2_lambda() is true, the value will be added to \a second_output_iter,
   * - otherwise the value will be added to \a unselected_iter.
   *
   * The output lists are in the same order as the input.
   *
   * After execution, it is possible to retrieve the number of elements
   * in the first and second lists using the \a nbParts() method.
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerThreeWayIf
   */
  template <typename InputIterator, typename FirstOutputIterator,
            typename SecondOutputIterator, typename UnselectedIterator,
            typename Select1Lambda, typename Select2Lambda>
  void applyIf(Int32 nb_value, InputIterator input_iter,
               FirstOutputIterator first_output_iter,
               SecondOutputIterator second_output_iter,
               UnselectedIterator unselected_iter,
               const Select1Lambda& select1_lambda,
               const Select2Lambda& select2_lambda,
               const TraceInfo& trace_info = TraceInfo())
  {
    if (_checkEmpty(nb_value))
      return;
    _setCalled();
    Impl::GenericPartitionerBase* base_ptr = this;
    Impl::GenericPartitionerIf gf;
    gf.apply3(*base_ptr, nb_value, input_iter, first_output_iter, second_output_iter,
              unselected_iter, select1_lambda, select2_lambda, trace_info);
  }

  /*!
   * \brief Number of elements in the first part of the list.
   */
  Int32 nbFirstPart()
  {
    m_is_already_called = false;
    return _nbFirstPart();
  }

  /*!
   * \brief Number of elements in the first and second parts of the list.
   *
   * Returns a view of two values. The first value contains the number
   * of elements in the first list and the second value the
   * number of elements in the second list.
   *
   * This method is only valid after calling a three-way partitioning method.
   */
  SmallSpan<const Int32> nbParts()
  {
    m_is_already_called = false;
    return _nbParts();
  }

 private:

  bool m_is_already_called = false;

 private:

  void _setCalled()
  {
    if (m_is_already_called)
      ARCCORE_FATAL("apply() has already been called for this instance");
    m_is_already_called = true;
  }
  bool _checkEmpty(Int32 nb_value)
  {
    if (nb_value == 0) {
      _resetNbPart();
      return true;
    }
    return false;
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
