// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Reduce.h                                                    (C) 2000-2026 */
/*                                                                           */
/* Reduction management for accelerators.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_REDUCE_H
#define ARCCORE_ACCELERATOR_REDUCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/String.h"
#include "arccore/base/FatalErrorException.h"

#include "arccore/common/accelerator/IReduceMemoryImpl.h"
#include "arccore/common/accelerator/RunCommandLaunchInfo.h"

#include "arccore/accelerator/CommonUtils.h"

#if defined(ARCCORE_COMPILING_SYCL)
#include "arccore/accelerator/RunCommandLoop.h"
#endif

#include <limits.h>
#include <float.h>
#include <atomic>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{
class HostDeviceReducerKernelRemainingArg;
}

namespace Arcane::impl
{
class HostReducerHelper;
}

namespace Arcane::Accelerator::Impl
{
class KernelReducerHelper;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command);

template <typename DataType>
class ReduceIdentity;
template <>
// TODO: use numeric_limits.
class ReduceIdentity<double>
{
 public:

  ARCCORE_HOST_DEVICE static constexpr double sumValue() { return 0.0; }
  ARCCORE_HOST_DEVICE static constexpr double minValue() { return DBL_MAX; }
  ARCCORE_HOST_DEVICE static constexpr double maxValue() { return -DBL_MAX; }
};
template <>
class ReduceIdentity<Int32>
{
 public:

  ARCCORE_HOST_DEVICE static constexpr Int32 sumValue() { return 0; }
  ARCCORE_HOST_DEVICE static constexpr Int32 minValue() { return INT32_MAX; }
  ARCCORE_HOST_DEVICE static constexpr Int32 maxValue() { return -INT32_MAX; }
};
template <>
class ReduceIdentity<Int64>
{
 public:

  ARCCORE_HOST_DEVICE static constexpr Int64 sumValue() { return 0; }
  ARCCORE_HOST_DEVICE static constexpr Int64 minValue() { return INT64_MAX; }
  ARCCORE_HOST_DEVICE static constexpr Int64 maxValue() { return -INT64_MAX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// The implementation used is defined in 'CommonCudaHipReduceImpl.h'

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Information to perform a reduction on a device.
 */
template <typename DataType>
class ReduceDeviceInfo
{
 public:

  //! Current thread value to reduce.
  DataType m_current_value = {};
  //! Pointer to the reduced data (HostPinned memory accessible from host
  //! and accelerator)
  DataType* m_host_pinned_final_ptr = nullptr;
  //! Array with a per-block value for the reduction
  SmallSpan<DataType> m_grid_buffer;
  /*!
   * Pointer to a memory region containing an integer to indicate
   * how many blocks remain to be reduced.
   * The associated memory is allocated on the accelerator.
   */
  unsigned int* m_device_count = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class ReduceAtomicSum;

template <>
class ReduceAtomicSum<double>
{
 public:

  static double apply(double* vptr, double v)
  {
    std::atomic_ref<double> aref(*vptr);
    double old = aref.load(std::memory_order_consume);
    double wanted = old + v;
    while (!aref.compare_exchange_weak(old, wanted, std::memory_order_release, std::memory_order_consume))
      wanted = old + v;
    return wanted;
  }
};
template <>
class ReduceAtomicSum<Int64>
{
 public:

  static Int64 apply(Int64* vptr, Int64 v)
  {
    std::atomic_ref<Int64> aref(*vptr);
    Int64 x = aref.fetch_add(v);
    return x + v;
  }
};
template <>
class ReduceAtomicSum<Int32>
{
 public:

  static Int32 apply(Int32* vptr, Int32 v)
  {
    std::atomic_ref<Int32> aref(*vptr);
    Int32 x = aref.fetch_add(v);
    return x + v;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType_>
class ReduceFunctorSum
{
 public:

  using DataType = DataType_;
  using ThatClass = ReduceFunctorSum<DataType>;

 public:

  static ARCCORE_DEVICE void
  applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
  {
    _applyDeviceGeneric<ThatClass>(dev_info);
  }
  static DataType applyAtomicOnHost(DataType* vptr, DataType v)
  {
    return ReduceAtomicSum<DataType>::apply(vptr, v);
  }

#if defined(ARCCORE_COMPILING_SYCL)
  static sycl::plus<DataType> syclFunctor() { return {}; }
#endif

  static ARCCORE_DEVICE inline void combine(DataType& val, const DataType v)
  {
    val = val + v;
  }

  ARCCORE_HOST_DEVICE static constexpr DataType identity() { return ReduceIdentity<DataType>::sumValue(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType_>
class ReduceFunctorMax
{
 public:

  using DataType = DataType_;
  using ThatClass = ReduceFunctorMax<DataType>;

 public:

  static ARCCORE_DEVICE void
  applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
  {
    _applyDeviceGeneric<ThatClass>(dev_info);
  }
  static DataType applyAtomicOnHost(DataType* ptr, DataType v)
  {
    std::atomic_ref<DataType> aref(*ptr);
    DataType prev_value = aref.load();
    while (prev_value < v && !aref.compare_exchange_weak(prev_value, v)) {
    }
    return aref.load();
  }
#if defined(ARCCORE_COMPILING_SYCL)
  static sycl::maximum<DataType> syclFunctor() { return {}; }
#endif

  static ARCCORE_DEVICE inline void combine(DataType& val, const DataType v)
  {
    val = v > val ? v : val;
  }

  ARCCORE_HOST_DEVICE static constexpr DataType identity() { return ReduceIdentity<DataType>::maxValue(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType_>
class ReduceFunctorMin
{
 public:

  using DataType = DataType_;
  using ThatClass = ReduceFunctorMin<DataType>;

 public:

  static ARCCORE_DEVICE void
  applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
  {
    _applyDeviceGeneric<ThatClass>(dev_info);
  }
  static DataType applyAtomicOnHost(DataType* vptr, DataType v)
  {
    std::atomic_ref<DataType> aref(*vptr);
    DataType prev_value = aref.load();
    while (prev_value > v && !aref.compare_exchange_weak(prev_value, v)) {
    }
    return aref.load();
  }
#if defined(ARCCORE_COMPILING_SYCL)
  static sycl::minimum<DataType> syclFunctor() { return {}; }
#endif

  static ARCCORE_DEVICE inline void combine(DataType& val, const DataType v)
  {
    val = v < val ? v : val;
  }

  ARCCORE_HOST_DEVICE static constexpr DataType identity() { return ReduceIdentity<DataType>::minValue(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reduction operator
 *
 * This class allows managing a reduction on an accelerator or in
 * multi-thread.
 *
 * The final reduction takes place when calling reduce(). Therefore, this call
 * must only be made once and in a collective part. This call is only
 * valid on instances created with an empty constructor. These latter
 * can only be created on the host.
 *
 * \warning The copy constructor must not be called explicitly.
 * The starting instance must remain valid as long as there are copies or
 * references in the computation kernel.
 *
 * NOTE on implementation
 *
 * On GPU, reductions are performed in the class destructor
 * The value 'm_host_or_device_memory_for_reduced_value' is used to retain these values.
 * On the host, an 'std::atomic' is used to maintain the common value
 * between threads. This value is referenced by 'm_parent_value' and is only
 * valid on the host.
 */
template <typename DataType, typename ReduceFunctor>
class HostDeviceReducerBase
{
 public:

  HostDeviceReducerBase(RunCommand& command)
  : m_host_memory_for_reduced_value(&m_local_value)
  {
    //std::cout << String::format("Reduce main host this={0}\n",this); std::cout.flush();
    m_is_master_instance = true;
    m_local_value = ReduceFunctor::identity();
    m_atomic_value = m_local_value;
    m_atomic_parent_value = &m_atomic_value;
    //printf("Create null host parent_value=%p this=%p\n",(void*)m_parent_value,(void*)this);
    m_memory_impl = Impl::internalGetOrCreateReduceMemoryImpl(&command);
    if (m_memory_impl) {
      m_memory_impl->allocateReduceDataMemory(sizeof(DataType));
      m_grid_memory_info = m_memory_impl->gridMemoryInfo();
      // Initialize the final value for the case where the reduction will not be
      // performed (e.g., if the RunCommand is never launched)
      DataType* ptr = _getHostPinnedMemoryForReducedValue();
      *ptr = m_local_value;
    }
  }

  // The Intel compiler considers this class not 'is_trivially_copyable'
  // on the device if the copy constructor is not used.
#if defined(__INTEL_LLVM_COMPILER) && defined(__SYCL_DEVICE_ONLY__)
  HostDeviceReducerBase(const HostDeviceReducerBase& rhs) = default;
#else
  ARCCORE_HOST_DEVICE HostDeviceReducerBase(const HostDeviceReducerBase& rhs)
  : m_host_memory_for_reduced_value(rhs.m_host_memory_for_reduced_value)
  , m_local_value(rhs.m_local_value)
  {
#ifdef ARCCORE_DEVICE_CODE
    m_grid_memory_info = rhs.m_grid_memory_info;
    //int threadId = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
    //if (threadId==0)
    //printf("Create ref device Id=%d parent=%p\n",threadId,&rhs);
#else
    m_memory_impl = rhs.m_memory_impl;
    if (m_memory_impl) {
      m_grid_memory_info = m_memory_impl->gridMemoryInfo();
    }
    //std::cout << String::format("Reduce: host copy this={0} rhs={1} mem={2} device_count={3}\n",this,&rhs,m_memory_impl,(void*)m_grid_device_count);
    m_atomic_parent_value = rhs.m_atomic_parent_value;
    m_local_value = ReduceFunctor::identity();
    m_atomic_value = m_local_value;
    //std::cout << String::format("Reduce copy host  this={0} parent_value={1} rhs={2}\n",this,(void*)m_parent_value,&rhs); std::cout.flush();
    //if (!rhs.m_is_master_instance)
    //ARCCORE_FATAL("Only copy from master instance is allowed");
    //printf("Create ref host parent_value=%p this=%p rhs=%p\n",(void*)m_parent_value,(void*)this,(void*)&rhs);
#endif
  }
#endif

  ARCCORE_HOST_DEVICE HostDeviceReducerBase(HostDeviceReducerBase&& rhs) = delete;
  HostDeviceReducerBase& operator=(const HostDeviceReducerBase& rhs) = delete;

 public:

  ARCCORE_HOST_DEVICE void setValue(DataType v)
  {
    m_local_value = v;
  }
  ARCCORE_HOST_DEVICE DataType localValue() const
  {
    return m_local_value;
  }

 protected:

  Impl::IReduceMemoryImpl* m_memory_impl = nullptr;
  /*!
   * \brief Pointer to the data that will contain the reduced value.
   *
   * This value is only valid if the reduction takes place on the host.
   */
  DataType* m_host_memory_for_reduced_value = nullptr;
  Impl::IReduceMemoryImpl::GridMemoryInfo m_grid_memory_info;

  mutable DataType m_local_value;
  DataType* m_atomic_parent_value = nullptr;
  mutable DataType m_atomic_value;

 private:

  bool m_is_master_instance = false;

 protected:

  //! Performs the reduction and retrieves the value. WARNING: only do this once.
  DataType _reduce()
  {
    if (!m_is_master_instance)
      ARCCORE_FATAL("Final reduce operation is only valid on master instance");

    DataType* final_ptr = m_host_memory_for_reduced_value;
    if (m_memory_impl) {
      final_ptr = _getHostPinnedMemoryForReducedValue();
      m_memory_impl->release();
      m_memory_impl = nullptr;
    }

    if (m_atomic_parent_value) {
      //std::cout << String::format("Reduce host has parent this={0} local_value={1} parent_value={2}\n",
      //                            this,m_local_value,*m_parent_value);
      //std::cout.flush();
      ReduceFunctor::applyAtomicOnHost(m_atomic_parent_value, *final_ptr);
      *final_ptr = *m_atomic_parent_value;
    }
    else {
      //std::cout << String::format("Reduce host no parent this={0} local_value={1} managed={2}\n",
      //                            this,m_local_value,*m_host_or_device_memory_for_reduced_value);
      //std::cout.flush();
    }
    return *final_ptr;
  }

  // NOTE: When the V1 version of the reduction is no longer available, this method will
  // only be called from the device.
  ARCCORE_HOST_DEVICE void
  _finalize()
  {
#if defined(ARCCORE_DEVICE_CODE)
    //int threadId = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
    //if ((threadId%16)==0)
    //printf("Destroy device Id=%d\n",threadId);
    auto buf_span = m_grid_memory_info.m_grid_memory_values.bytes();
    DataType* buf = reinterpret_cast<DataType*>(buf_span.data());
    SmallSpan<DataType> grid_buffer(buf, static_cast<Int32>(buf_span.size()));

    Impl::ReduceDeviceInfo<DataType> dvi;
    dvi.m_grid_buffer = grid_buffer;
    dvi.m_device_count = m_grid_memory_info.m_grid_device_count;
    dvi.m_host_pinned_final_ptr = _getHostPinnedMemoryForReducedValue();
    dvi.m_current_value = m_local_value;
#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)
    ReduceFunctor::applyDevice(dvi); //grid_buffer,m_grid_device_count,m_host_or_device_memory_for_reduced_value,m_local_value,m_identity);
#endif
#else
    //      printf("Destroy host parent_value=%p this=%p\n",(void*)m_parent_value,(void*)this);
    // Host code
    //std::cout << String::format("Reduce destructor this={0} parent_value={1} v={2} memory_impl={3}\n",this,(void*)m_parent_value,m_local_value,m_memory_impl);
    //std::cout << String::format("Reduce destructor this={0} grid_data={1} grid_size={2}\n",
    //                            this,(void*)m_grid_memory_value_as_bytes,m_grid_memory_size);
    //std::cout.flush();
    if (!m_is_master_instance)
      ReduceFunctor::applyAtomicOnHost(m_atomic_parent_value, m_local_value);

    //printf("Destroy host %p %p\n",m_host_or_device_memory_for_reduced_value,this);
#endif
  }

 private:

  /*!
   * \brief Memory zone that will contain the reduction result if it
   * is performed on the Device.
   */
  ARCCORE_HOST_DEVICE DataType* _getHostPinnedMemoryForReducedValue()
  {
    return reinterpret_cast<DataType*>(m_grid_memory_info.m_host_memory_for_reduced_value);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Version 1 of the reduction.
 *
 * This version is obsolete. It uses the class destructor
 * to perform the reduction.
 */
template <typename DataType, typename ReduceFunctor>
class HostDeviceReducer
: public HostDeviceReducerBase<DataType, ReduceFunctor>
{
 public:

  using BaseClass = HostDeviceReducerBase<DataType, ReduceFunctor>;

 public:

  explicit HostDeviceReducer(RunCommand& command)
  : BaseClass(command)
  {}
  HostDeviceReducer(const HostDeviceReducer& rhs) = default;
  ARCCORE_HOST_DEVICE ~HostDeviceReducer()
  {
    this->_finalize();
  }

 public:

  DataType reduce()
  {
    return this->_reduce();
  }

  DataType reducedValue()
  {
    return this->_reduce();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Version 2 of the reduction.
 */
template <typename DataType, typename ReduceFunctor>
class HostDeviceReducer2
: public HostDeviceReducerBase<DataType, ReduceFunctor>
{
  friend Impl::HostDeviceReducerKernelRemainingArg;

 public:

  using BaseClass = HostDeviceReducerBase<DataType, ReduceFunctor>;
  using BaseClass::m_grid_memory_info;
  using BaseClass::m_host_memory_for_reduced_value;
  using BaseClass::m_local_value;

  using RemainingArgHandlerType = Impl::HostDeviceReducerKernelRemainingArg;

 public:

  explicit HostDeviceReducer2(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  DataType reducedValue()
  {
    return this->_reduce();
  }

 private:


#if defined(ARCCORE_COMPILING_SYCL)
  void _internalExecWorkItemAtEnd(sycl::nd_item<1> id)
  {
    unsigned int* atomic_counter_ptr = m_grid_memory_info.m_grid_device_count;
    const Int32 local_id = static_cast<Int32>(id.get_local_id(0));
    const Int32 group_id = static_cast<Int32>(id.get_group_linear_id());
    const Int32 nb_block = static_cast<Int32>(id.get_group_range(0));

    auto buf_span = m_grid_memory_info.m_grid_memory_values.bytes();
    DataType* buf = reinterpret_cast<DataType*>(buf_span.data());
    SmallSpan<DataType> grid_buffer(buf, static_cast<Int32>(buf_span.size()));

    DataType v = m_local_value;
    bool is_last = false;
    auto sycl_functor = ReduceFunctor::syclFunctor();
    DataType local_sum = sycl::reduce_over_group(id.get_group(), v, sycl_functor);
    if (local_id == 0) {
      grid_buffer[group_id] = local_sum;

      // TODO: In theory, one should perform the equivalent of a __threadfence() here
      // to ensure that other work-items see the update to 'grid_buffer'.
      // But this mechanism does not exist with SYCL 2020.

      // AdaptiveCpp 2024.2 does not support atomic operations on 'unsigned int'.
      // They are supported with the 'int' type. Since we are certain not to exceed 2^31, we
      // convert the pointer to an 'int*'.
#if defined(__ADAPTIVECPP__)
      int* atomic_counter_ptr_as_int = reinterpret_cast<int*>(atomic_counter_ptr);
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*atomic_counter_ptr_as_int);
#else
      sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*atomic_counter_ptr);
#endif
      Int32 cx = a.fetch_add(1);
      if (cx == (nb_block - 1))
        is_last = true;
    }

    // I am the last one to perform the reduction.
    // Calculate the final reduction
    if (is_last) {
      DataType my_total = grid_buffer[0];
      for (int x = 1; x < nb_block; ++x)
        my_total = sycl_functor(my_total, grid_buffer[x]);
      // Put the final result in the first element of the array.
      grid_buffer[0] = my_total;
      DataType* final_value_ptr = reinterpret_cast<DataType*>(m_grid_memory_info.m_host_memory_for_reduced_value);
      *final_value_ptr = my_total;
      *atomic_counter_ptr = 0;
    }
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of the reduction for the SYCL backend.
 *
 * \warning Currently there is no implementation. This class only allows
 * compilation.
 */
template <typename DataType, typename ReduceFunctor>
class SyclReducer
{
 public:

  explicit SyclReducer(RunCommand&) {}

 public:

  DataType reduce()
  {
    return m_local_value;
  }
  void setValue(DataType v) { m_local_value = v; }

 protected:

  mutable DataType m_local_value = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)
template <typename DataType, typename ReduceFunctor> using Reducer = SyclReducer<DataType, ReduceFunctor>;
#else
template <typename DataType, typename ReduceFunctor> using Reducer = HostDeviceReducer<DataType, ReduceFunctor>;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to perform a 'sum' reduction.
 */
template <typename DataType>
class ReducerSum
: public Reducer<DataType, Impl::ReduceFunctorSum<DataType>>
{
  using BaseClass = Reducer<DataType, Impl::ReduceFunctorSum<DataType>>;
  using BaseClass::m_local_value;

 public:

  explicit ReducerSum(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE DataType combine(DataType v) const
  {
    m_local_value += v;
    return m_local_value;
  }

  ARCCORE_HOST_DEVICE DataType add(DataType v) const
  {
    return combine(v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to perform a 'max' reduction.
 */
template <typename DataType>
class ReducerMax
: public Reducer<DataType, Impl::ReduceFunctorMax<DataType>>
{
  using BaseClass = Reducer<DataType, Impl::ReduceFunctorMax<DataType>>;
  using BaseClass::m_local_value;

 public:

  explicit ReducerMax(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE DataType combine(DataType v) const
  {
    m_local_value = v > m_local_value ? v : m_local_value;
    return m_local_value;
  }

  ARCCORE_HOST_DEVICE DataType max(DataType v) const
  {
    return combine(v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to perform a 'min' reduction.
 */
template <typename DataType>
class ReducerMin
: public Reducer<DataType, Impl::ReduceFunctorMin<DataType>>
{
  using BaseClass = Reducer<DataType, Impl::ReduceFunctorMin<DataType>>;
  using BaseClass::m_local_value;

 public:

  explicit ReducerMin(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE DataType combine(DataType v) const
  {
    m_local_value = v < m_local_value ? v : m_local_value;
    return m_local_value;
  }

  ARCCORE_HOST_DEVICE DataType min(DataType v) const
  {
    return combine(v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to perform a 'sum' reduction.
 */
template <typename DataType>
class ReducerSum2
: public HostDeviceReducer2<DataType, Impl::ReduceFunctorSum<DataType>>
{
  using BaseClass = HostDeviceReducer2<DataType, Impl::ReduceFunctorSum<DataType>>;

 public:

  explicit ReducerSum2(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE void combine(DataType v)
  {
    this->m_local_value += v;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to perform a 'max' reduction.
 */
template <typename DataType>
class ReducerMax2
: public HostDeviceReducer2<DataType, Impl::ReduceFunctorMax<DataType>>
{
  using BaseClass = HostDeviceReducer2<DataType, Impl::ReduceFunctorMax<DataType>>;

 public:

  explicit ReducerMax2(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE void combine(DataType v)
  {
    DataType& lv = this->m_local_value;
    lv = v > lv ? v : lv;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to perform a 'min' reduction.
 */
template <typename DataType>
class ReducerMin2
: public HostDeviceReducer2<DataType, Impl::ReduceFunctorMin<DataType>>
{
  using BaseClass = HostDeviceReducer2<DataType, Impl::ReduceFunctorMin<DataType>>;

 public:

  explicit ReducerMin2(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE void combine(DataType v)
  {
    DataType& lv = this->m_local_value;
    lv = v < lv ? v : lv;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to manage HostDeviceReducer2 arguments at the beginning
 * and end of kernel execution.
 */
class Impl::HostDeviceReducerKernelRemainingArg
{
 public:

  template <typename DataType, typename ReduceFunctor>
  static bool isNeedBarrier(const HostDeviceReducer2<DataType, ReduceFunctor>&)
  {
    return true;
  }

  template <typename DataType, typename ReduceFunctor>
  static void
  execWorkItemAtBeginForHost(HostDeviceReducer2<DataType, ReduceFunctor>&)
  {
  }
  template <typename DataType, typename ReduceFunctor>
  static void
  execWorkItemAtEndForHost(HostDeviceReducer2<DataType, ReduceFunctor>& reducer)
  {
    reducer._finalize();
  }

  template <typename DataType, typename ReduceFunctor>
  static ARCCORE_DEVICE void
  execWorkItemAtBeginForCudaHip(HostDeviceReducer2<DataType, ReduceFunctor>&, Int32)
  {
  }

  template <typename DataType, typename ReduceFunctor>
  static ARCCORE_DEVICE void
  execWorkItemAtEndForCudaHip(HostDeviceReducer2<DataType, ReduceFunctor>& reducer, Int32)
  {
    reducer._finalize();
  }

#if defined(ARCCORE_COMPILING_SYCL)
  template <typename DataType, typename ReduceFunctor>
  static void
  execWorkItemAtBeginForSycl(HostDeviceReducer2<DataType, ReduceFunctor>&, sycl::nd_item<1>)
  {
  }
  template <typename DataType, typename ReduceFunctor>
  static void
  execWorkItemAtEndForSycl(HostDeviceReducer2<DataType, ReduceFunctor>& reducer, sycl::nd_item<1> id)
  {
    reducer._internalExecWorkItemAtEnd(id);
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// This macro is defined if we want to make the implementation inline.
// Ideally, this should not be the case (which would allow changing the
// implementation without recompiling everything) but it doesn't seem to
// work well for now.

#define ARCCORE_INLINE_REDUCE_IMPL

#ifdef ARCCORE_INLINE_REDUCE_IMPL

#  ifndef ARCCORE_INLINE_REDUCE
#    define ARCCORE_INLINE_REDUCE inline
#  endif

#if defined(__CUDACC__) || defined(__HIP__)
#  include "arccore/accelerator/CommonCudaHipReduceImpl.h"
#else

#endif

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
