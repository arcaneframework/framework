// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Reduce.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Gestion des réductions pour les accélérateurs.                            */
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

namespace Arcane::Accelerator::impl
{
class KernelReducerHelper;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command);

template <typename DataType>
class ReduceIdentity;
template <>
// TODO: utiliser numeric_limits.
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
// L'implémentation utilisée est définie dans 'CommonCudaHipReduceImpl.h'

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations pour effectuer une réduction sur un device.
 */
template <typename DataType>
class ReduceDeviceInfo
{
 public:

  //! Valeur du thread courant à réduire.
  DataType m_current_value = {};
  //! Pointeur vers la donnée réduite (mémoire HostPinned accessible depuis l'hôte et l'accélérateur)
  DataType* m_host_pinned_final_ptr = nullptr;
  //! Tableau avec une valeur par bloc pour la réduction
  SmallSpan<DataType> m_grid_buffer;
  /*!
   * Pointeur vers une zone mémoire contenant un entier pour indiquer
   * combien il reste de blocs à réduire.
   * La mémoire associée est allouée sur l'accélérateur.
   */
  unsigned int* m_device_count = nullptr;

  //! Taille d'un warp
  Int32 m_warp_size = 0;
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

template <typename DataType>
class ReduceFunctorSum
{
 public:

  static ARCCORE_DEVICE void
  applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
  {
    _applyDevice(dev_info);
  }
  static DataType apply(DataType* vptr, DataType v)
  {
    return ReduceAtomicSum<DataType>::apply(vptr, v);
  }
#if defined(ARCCORE_COMPILING_SYCL)
  static sycl::plus<DataType> syclFunctor() { return {}; }
#endif

 public:

  ARCCORE_HOST_DEVICE static constexpr DataType identity() { return impl::ReduceIdentity<DataType>::sumValue(); }

 private:

  static ARCCORE_DEVICE void _applyDevice(const ReduceDeviceInfo<DataType>& dev_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class ReduceFunctorMax
{
 public:

  static ARCCORE_DEVICE void
  applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
  {
    _applyDevice(dev_info);
  }
  static DataType apply(DataType* ptr, DataType v)
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

 public:

  ARCCORE_HOST_DEVICE static constexpr DataType identity() { return impl::ReduceIdentity<DataType>::maxValue(); }

 private:

  static ARCCORE_DEVICE void _applyDevice(const ReduceDeviceInfo<DataType>& dev_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class ReduceFunctorMin
{
 public:

  static ARCCORE_DEVICE void
  applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
  {
    _applyDevice(dev_info);
  }
  static DataType apply(DataType* vptr, DataType v)
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

 public:

  ARCCORE_HOST_DEVICE static constexpr DataType identity() { return impl::ReduceIdentity<DataType>::minValue(); }

 private:

  static ARCCORE_DEVICE void _applyDevice(const ReduceDeviceInfo<DataType>& dev_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérateur de réduction
 *
 * Cette classe permet de gérer une réduction sur accélérateur ou en
 * multi-thread.
 *
 * La réduction finale a lieu lors de l'appel à reduce(). Il ne faut donc
 * faire cet appel qu'une seule fois et dans une partie collective. Cet appel
 * n'est valide que sur les instance créées avec un constructeur vide. Ces dernières
 * ne peuvent être créées que sur l'hôte.
 *
 * \warning Le constructeur de recopie ne doit pas être appelé explicitement.
 * L'instance de départ doit rester valide tant qu'il existe des copies ou
 * des références dans le noyau de calcul.
 *
 * NOTE sur l'implémentation
 *
 * Sur GPU, les réductions sont effectuées dans le destructeur de la classe
 * La valeur 'm_host_or_device_memory_for_reduced_value' sert à conserver ces valeurs.
 * Sur l'hôte, on utilise un 'std::atomic' pour conserver la valeur commune
 * entre les threads. Cette valeur est référencée par 'm_parent_value' et n'est
 * valide que sur l'hôte.
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
    m_memory_impl = impl::internalGetOrCreateReduceMemoryImpl(&command);
    if (m_memory_impl) {
      m_memory_impl->allocateReduceDataMemory(sizeof(DataType));
      m_grid_memory_info = m_memory_impl->gridMemoryInfo();
    }
  }

  // Le compilateur Intel considère que cette classe n'est pas 'is_trivially_copyable'
  // sur le device si on n'utilise pas le constructeur de copie.
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

  impl::IReduceMemoryImpl* m_memory_impl = nullptr;
  /*!
   * \brief Pointeur vers la donnée qui contiendra la valeur réduite.
   *
   * Cette valeur est uniquement valide si la réduction a lieu sur l'hôte.
   */
  DataType* m_host_memory_for_reduced_value = nullptr;
  impl::IReduceMemoryImpl::GridMemoryInfo m_grid_memory_info;

  mutable DataType m_local_value;
  DataType* m_atomic_parent_value = nullptr;
  mutable DataType m_atomic_value;

 private:

  bool m_is_master_instance = false;

 protected:

  //! Effectue la réduction et récupère la valeur. ATTENTION: ne faire qu'une seule fois.
  DataType _reduce()
  {
    if (!m_is_master_instance)
      ARCCORE_FATAL("Final reduce operation is only valid on master instance");

    DataType* final_ptr = m_host_memory_for_reduced_value;
    if (m_memory_impl) {
      final_ptr = reinterpret_cast<DataType*>(m_grid_memory_info.m_host_memory_for_reduced_value);
      m_memory_impl->release();
      m_memory_impl = nullptr;
    }

    if (m_atomic_parent_value) {
      //std::cout << String::format("Reduce host has parent this={0} local_value={1} parent_value={2}\n",
      //                            this,m_local_value,*m_parent_value);
      //std::cout.flush();
      ReduceFunctor::apply(m_atomic_parent_value, *final_ptr);
      *final_ptr = *m_atomic_parent_value;
    }
    else {
      //std::cout << String::format("Reduce host no parent this={0} local_value={1} managed={2}\n",
      //                            this,m_local_value,*m_host_or_device_memory_for_reduced_value);
      //std::cout.flush();
    }
    return *final_ptr;
  }

  // NOTE: Lorsqu'il n'y aura plus la version V1 de la réduction, cette méthode ne sera
  // appelée que depuis le device.
  ARCCORE_HOST_DEVICE void
  _finalize()
  {
#ifdef ARCCORE_DEVICE_CODE
    //int threadId = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
    //if ((threadId%16)==0)
    //printf("Destroy device Id=%d\n",threadId);
    auto buf_span = m_grid_memory_info.m_grid_memory_values.bytes();
    DataType* buf = reinterpret_cast<DataType*>(buf_span.data());
    SmallSpan<DataType> grid_buffer(buf, static_cast<Int32>(buf_span.size()));

    impl::ReduceDeviceInfo<DataType> dvi;
    dvi.m_grid_buffer = grid_buffer;
    dvi.m_device_count = m_grid_memory_info.m_grid_device_count;
    dvi.m_host_pinned_final_ptr = reinterpret_cast<DataType*>(m_grid_memory_info.m_host_memory_for_reduced_value);
    dvi.m_current_value = m_local_value;
    dvi.m_warp_size = m_grid_memory_info.m_warp_size;
    ReduceFunctor::applyDevice(dvi); //grid_buffer,m_grid_device_count,m_host_or_device_memory_for_reduced_value,m_local_value,m_identity);
#else
    //      printf("Destroy host parent_value=%p this=%p\n",(void*)m_parent_value,(void*)this);
    // Code hôte
    //std::cout << String::format("Reduce destructor this={0} parent_value={1} v={2} memory_impl={3}\n",this,(void*)m_parent_value,m_local_value,m_memory_impl);
    //std::cout << String::format("Reduce destructor this={0} grid_data={1} grid_size={2}\n",
    //                            this,(void*)m_grid_memory_value_as_bytes,m_grid_memory_size);
    //std::cout.flush();
    if (!m_is_master_instance)
      ReduceFunctor::apply(m_atomic_parent_value, m_local_value);

    //printf("Destroy host %p %p\n",m_host_or_device_memory_for_reduced_value,this);
#endif
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Version 1 de la réduction.
 *
 * Cette version est obsolète. Elle utilise le destructeur de la classe
 * pour effectuer la réduction.
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
 * \brief Version 2 de la réduction.
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

      // TODO: En théorie il faut faire l'équivalent d'un __threadfence() ici
      // pour garantir que les autres work-item voient bien la mise à jour de 'grid_buffer'.
      // Mais ce mécanisme n'existe pas avec SYCL 2020.

      // AdaptiveCpp 2024.2 ne supporte pas les opérations atomiques sur 'unsigned int'.
      // Elles sont supportées avec le type 'int'. Comme on est certain de ne pas dépasser 2^31, on
      // converti le pointeur en un 'int*'.
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

    // Je suis le dernier à faire la réduction.
    // Calcule la réduction finale
    if (is_last) {
      DataType my_total = grid_buffer[0];
      for (int x = 1; x < nb_block; ++x)
        my_total = sycl_functor(my_total, grid_buffer[x]);
      // Met le résultat final dans le premier élément du tableau.
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
 * \brief Implémentation de la réduction pour le backend SYCL.
 *
 * \warning Pour l'instant il n'y aucune implémentation. Cette classe permet
 * juste la compilation.
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
 * \brief Classe pour effectuer une réduction 'somme'.
 */
template <typename DataType>
class ReducerSum
: public Reducer<DataType, impl::ReduceFunctorSum<DataType>>
{
  using BaseClass = Reducer<DataType, impl::ReduceFunctorSum<DataType>>;
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
 * \brief Classe pour effectuer une réduction 'max'.
 */
template <typename DataType>
class ReducerMax
: public Reducer<DataType, impl::ReduceFunctorMax<DataType>>
{
  using BaseClass = Reducer<DataType, impl::ReduceFunctorMax<DataType>>;
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
 * \brief Classe pour effectuer une réduction 'min'.
 */
template <typename DataType>
class ReducerMin
: public Reducer<DataType, impl::ReduceFunctorMin<DataType>>
{
  using BaseClass = Reducer<DataType, impl::ReduceFunctorMin<DataType>>;
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
 * \brief Classe pour effectuer une réduction 'somme'.
 */
template <typename DataType>
class ReducerSum2
: public HostDeviceReducer2<DataType, impl::ReduceFunctorSum<DataType>>
{
  using BaseClass = HostDeviceReducer2<DataType, impl::ReduceFunctorSum<DataType>>;

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
 * \brief Classe pour effectuer une réduction 'max'.
 */
template <typename DataType>
class ReducerMax2
: public HostDeviceReducer2<DataType, impl::ReduceFunctorMax<DataType>>
{
  using BaseClass = HostDeviceReducer2<DataType, impl::ReduceFunctorMax<DataType>>;

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
 * \brief Classe pour effectuer une réduction 'min'.
 */
template <typename DataType>
class ReducerMin2
: public HostDeviceReducer2<DataType, impl::ReduceFunctorMin<DataType>>
{
  using BaseClass = HostDeviceReducer2<DataType, impl::ReduceFunctorMin<DataType>>;

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
 * \brief Classe pour gérer les arguments de type HostDeviceReducer2 en
 * début et fin d'exécution des noyaux.
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
// Cette macro est définie si on souhaite rendre inline l'implémentation.
// Dans l'idéal il ne faut pas que ce soit le cas (ce qui permettrait de
// changer l'implémentation sans tout recompiler) mais cela ne semble pas
// bien fonctionner pour l'instant.

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
