﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Reduce.h                                                    (C) 2000-2024 */
/*                                                                           */
/* Gestion des réductions pour les accélérateurs.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_REDUCE_H
#define ARCANE_ACCELERATOR_REDUCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

#include "arcane/accelerator/core/IReduceMemoryImpl.h"
#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/CommonUtils.h"

#include <limits.h>
#include <float.h>
#include <atomic>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
  extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IReduceMemoryImpl*
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
    DataType m_current_value;
    //! Valeur de l'identité pour la réduction
    DataType m_identity;
    //! Pointeur vers la donnée réduite (mémoire uniquement accessible depuis le device)
    DataType* m_device_final_ptr = nullptr;
    //! Pointeur vers la donnée réduite (mémoire uniquement accessible depuis l'hôte)
    void* m_host_final_ptr = nullptr;
    //! Tableau avec une valeur par bloc pour la réduction
    SmallSpan<DataType> m_grid_buffer;
    /*!
   * Pointeur vers une zone mémoire contenant un entier pour indiquer
   * combien il reste de blocs à réduire.
   */
    unsigned int* m_device_count = nullptr;

    //! Indique si on utilise la réduction par grille (sinon on utilise les atomiques)
    bool m_use_grid_reduce = true;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename DataType>
  class ReduceAtomicSum;

  template <>
  class ReduceAtomicSum<double>
  {
   public:

    static double apply(std::atomic<double>* ptr, double v)
    {
      double old = ptr->load(std::memory_order_consume);
      double wanted = old + v;
      while (!ptr->compare_exchange_weak(old, wanted, std::memory_order_release, std::memory_order_consume))
        wanted = old + v;
      return wanted;
    }
  };
  template <>
  class ReduceAtomicSum<Int64>
  {
   public:

    static Int64 apply(std::atomic<Int64>* ptr, Int64 v)
    {
      Int64 x = std::atomic_fetch_add(ptr, v);
      return x + v;
    }
  };
  template <>
  class ReduceAtomicSum<Int32>
  {
   public:

    static Int32 apply(std::atomic<Int32>* ptr, Int32 v)
    {
      Int32 x = std::atomic_fetch_add(ptr, v);
      return x + v;
    }
  };

  template <typename DataType>
  class ReduceFunctorSum
  {
   public:

    static ARCCORE_DEVICE
    DataType
    applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
    {
      _applyDevice(dev_info);
      return *(dev_info.m_device_final_ptr);
    }
    static DataType apply(std::atomic<DataType>* ptr, DataType v)
    {
      return ReduceAtomicSum<DataType>::apply(ptr, v);
    }

   public:

    ARCCORE_HOST_DEVICE static constexpr DataType identity() { return impl::ReduceIdentity<DataType>::sumValue(); }

   private:

    static ARCCORE_DEVICE void _applyDevice(const ReduceDeviceInfo<DataType>& dev_info);
  };

  template <typename DataType>
  class ReduceFunctorMax
  {
   public:

    static ARCCORE_DEVICE
    DataType
    applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
    {
      _applyDevice(dev_info);
      return *(dev_info.m_device_final_ptr);
    }
    static DataType apply(std::atomic<DataType>* ptr, DataType v)
    {
      DataType prev_value = *ptr;
      while (prev_value < v && !ptr->compare_exchange_weak(prev_value, v)) {
      }
      return *ptr;
    }

   public:

    ARCCORE_HOST_DEVICE static constexpr DataType identity() { return impl::ReduceIdentity<DataType>::maxValue(); }

   private:

    static ARCCORE_DEVICE void _applyDevice(const ReduceDeviceInfo<DataType>& dev_info);
  };

  template <typename DataType>
  class ReduceFunctorMin
  {
   public:

    static ARCCORE_DEVICE
    DataType
    applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
    {
      _applyDevice(dev_info);
      return *(dev_info.m_device_final_ptr);
    }
    static DataType apply(std::atomic<DataType>* ptr, DataType v)
    {
      DataType prev_value = *ptr;
      while (prev_value > v && !ptr->compare_exchange_weak(prev_value, v)) {
      }
      return *ptr;
    }

   public:

    ARCCORE_HOST_DEVICE static constexpr DataType identity() { return impl::ReduceIdentity<DataType>::minValue(); }

   private:

    static ARCCORE_DEVICE void _applyDevice(const ReduceDeviceInfo<DataType>& dev_info);
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // End namespace impl

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
class HostDeviceReducer
{
 public:

  HostDeviceReducer(RunCommand& command)
  : m_host_or_device_memory_for_reduced_value(&m_local_value)
  , m_command(&command)
  {
    //std::cout << String::format("Reduce main host this={0}\n",this); std::cout.flush();
    m_is_master_instance = true;
    m_identity = ReduceFunctor::identity();
    m_local_value = m_identity;
    m_atomic_value = m_identity;
    m_parent_value = &m_atomic_value;
    //printf("Create null host parent_value=%p this=%p\n",(void*)m_parent_value,(void*)this);
    m_memory_impl = impl::internalGetOrCreateReduceMemoryImpl(&command);
    if (m_memory_impl) {
      m_host_or_device_memory_for_reduced_value = impl::allocateReduceDataMemory<DataType>(m_memory_impl, m_identity);
      m_grid_memory_info = m_memory_impl->gridMemoryInfo();
    }
  }
  ARCCORE_HOST_DEVICE HostDeviceReducer(const HostDeviceReducer& rhs)
  : m_host_or_device_memory_for_reduced_value(rhs.m_host_or_device_memory_for_reduced_value)
  , m_local_value(rhs.m_local_value)
  , m_identity(rhs.m_identity)
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
    m_parent_value = rhs.m_parent_value;
    m_local_value = rhs.m_identity;
    m_atomic_value = m_identity;
    //std::cout << String::format("Reduce copy host  this={0} parent_value={1} rhs={2}\n",this,(void*)m_parent_value,&rhs); std::cout.flush();
    //if (!rhs.m_is_master_instance)
    //ARCANE_FATAL("Only copy from master instance is allowed");
    //printf("Create ref host parent_value=%p this=%p rhs=%p\n",(void*)m_parent_value,(void*)this,(void*)&rhs);
#endif
  }

  ARCCORE_HOST_DEVICE HostDeviceReducer(HostDeviceReducer&& rhs) = delete;
  HostDeviceReducer& operator=(const HostDeviceReducer& rhs) = delete;

  ARCCORE_HOST_DEVICE ~HostDeviceReducer()
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
    dvi.m_device_final_ptr = m_host_or_device_memory_for_reduced_value;
    dvi.m_host_final_ptr = m_grid_memory_info.m_host_memory_for_reduced_value;
    dvi.m_current_value = m_local_value;
    dvi.m_identity = m_identity;
    dvi.m_use_grid_reduce = m_grid_memory_info.m_reduce_policy != eDeviceReducePolicy::Atomic;
    ReduceFunctor::applyDevice(dvi); //grid_buffer,m_grid_device_count,m_host_or_device_memory_for_reduced_value,m_local_value,m_identity);
#else
    //      printf("Destroy host parent_value=%p this=%p\n",(void*)m_parent_value,(void*)this);
    // Code hôte
    //std::cout << String::format("Reduce destructor this={0} parent_value={1} v={2} memory_impl={3}\n",this,(void*)m_parent_value,m_local_value,m_memory_impl);
    //std::cout << String::format("Reduce destructor this={0} grid_data={1} grid_size={2}\n",
    //                            this,(void*)m_grid_memory_value_as_bytes,m_grid_memory_size);
    //std::cout.flush();
    if (!m_is_master_instance)
      ReduceFunctor::apply(m_parent_value, m_local_value);

    //printf("Destroy host %p %p\n",m_host_or_device_memory_for_reduced_value,this);
    if (m_memory_impl && m_is_master_instance)
      m_memory_impl->release();
#endif
  }

 public:

  ARCCORE_HOST_DEVICE void setValue(DataType v)
  {
    m_local_value = v;
  }
  ARCCORE_HOST_DEVICE DataType localValue() const
  {
    return m_local_value;
  }
  //! Effectue la réduction et récupère la valeur. ATTENTION: ne faire qu'une seule fois.
  DataType reduce()
  {
    // Si la réduction est faite sur accélérateur, il faut recopier la valeur du device sur l'hôte.
    DataType* final_ptr = m_host_or_device_memory_for_reduced_value;
    if (m_memory_impl) {
      m_memory_impl->copyReduceValueFromDevice();
      final_ptr = reinterpret_cast<DataType*>(m_grid_memory_info.m_host_memory_for_reduced_value);
    }

    if (m_parent_value) {
      //std::cout << String::format("Reduce host has parent this={0} local_value={1} parent_value={2}\n",
      //                            this,m_local_value,*m_parent_value);
      //std::cout.flush();
      ReduceFunctor::apply(m_parent_value, *final_ptr);
      *final_ptr = *m_parent_value;
    }
    else {
      //std::cout << String::format("Reduce host no parent this={0} local_value={1} managed={2}\n",
      //                            this,m_local_value,*m_host_or_device_memory_for_reduced_value);
      //std::cout.flush();
    }
    return *final_ptr;
  }

 protected:

  impl::IReduceMemoryImpl* m_memory_impl = nullptr;
  /*!
   * \brief Pointeur vers la donnée qui contiendra la valeur réduite.
   *
   * Sur accélérateur, cette donnée est allouée sur le device.
   * Sur CPU, il s'agit de l'adresse de \a m_local_value pour l'instance parente.
   */
  DataType* m_host_or_device_memory_for_reduced_value = nullptr;
  impl::IReduceMemoryImpl::GridMemoryInfo m_grid_memory_info;

 private:

  RunCommand* m_command = nullptr;

 protected:

  mutable DataType m_local_value;
  std::atomic<DataType>* m_parent_value = nullptr;
  mutable std::atomic<DataType> m_atomic_value;

 private:

  DataType m_identity;

 private:

  bool m_is_allocated = false;
  bool m_is_master_instance = false;
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

  SyclReducer(RunCommand&) {}

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

#if defined(ARCANE_COMPILING_SYCL)
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

  ReducerSum(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE DataType add(DataType v) const
  {
    m_local_value += v;
    return m_local_value;
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

  ReducerMax(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE DataType max(DataType v) const
  {
    m_local_value = v > m_local_value ? v : m_local_value;
    return m_local_value;
  }
};

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

  ReducerMin(RunCommand& command)
  : BaseClass(command)
  {}

 public:

  ARCCORE_HOST_DEVICE DataType min(DataType v) const
  {
    m_local_value = v < m_local_value ? v : m_local_value;
    return m_local_value;
  }
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

#define ARCANE_INLINE_REDUCE_IMPL

#ifdef ARCANE_INLINE_REDUCE_IMPL

#  ifndef ARCANE_INLINE_REDUCE
#    define ARCANE_INLINE_REDUCE inline
#  endif

#if defined(__CUDACC__) || defined(__HIP__)
#  include "arcane/accelerator/CommonCudaHipReduceImpl.h"
#else

#endif

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
template <typename DataType>
class GenericReducerIf;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base pour effectuer une réduction.
 *
 * Contient les arguments nécessaires pour effectuer une réduction.
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
    eMemoryRessource r = eMemoryRessource::Host;
    if (m_queue.isAcceleratorPolicy())
      r = eMemoryRessource::HostPinned;
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
 * \brief Classe pour effectuer un partitionnement d'une liste.
 *
 * La liste est partitionnée en deux listes.
 *
 * \a DataType est le type de donnée.
 */
template <typename DataType>
class GenericReducerIf
{
  // TODO: Faire le malloc sur le device associé à la queue.
  //       et aussi regarder si on peut utiliser mallocAsync().

 public:

  template <typename InputIterator, typename ReduceOperator>
  void apply(GenericReducerBase<DataType>& s, Int32 nb_item, const DataType& init_value,
             InputIterator input_iter, ReduceOperator reduce_op)
  {
    RunQueue& queue = s.m_queue;
    eExecutionPolicy exec_policy = queue.executionPolicy();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(&queue);
      DataType* reduced_value_ptr = nullptr;
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_CUDA(::cub::DeviceReduce::Reduce(nullptr, temp_storage_size, input_iter, reduced_value_ptr,
                                                    nb_item, reduce_op, init_value, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      reduced_value_ptr = s.m_device_reduce_storage.allocate();
      ARCANE_CHECK_CUDA(::cub::DeviceReduce::Reduce(s.m_algo_storage.address(), temp_storage_size,
                                                    input_iter, reduced_value_ptr, nb_item,
                                                    reduce_op, init_value, stream));
      s.m_device_reduce_storage.copyToAsync(s.m_host_reduce_storage, &queue);
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      hipStream_t stream = impl::HipUtils::toNativeStream(&queue);
      DataType* reduced_value_ptr = nullptr;
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_HIP(rocprim::reduce(nullptr, temp_storage_size, input_iter, reduced_value_ptr, init_value,
                                       nb_item, reduce_op, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      reduced_value_ptr = s.m_device_reduce_storage.allocate();

      ARCANE_CHECK_HIP(rocprim::reduce(s.m_algo_storage.address(), temp_storage_size, input_iter, reduced_value_ptr, init_value,
                                       nb_item, reduce_op, stream));
      s.m_device_reduce_storage.copyToAsync(s.m_host_reduce_storage, &queue);
    }
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
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
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithme générique de réduction sur accélérateur.
 *
 * La réduction se fait via les appels à applyMin(), applyMax(), applySum(),
 * applyMinWithIndex(), applyMaxWithIndex() ou applySumWithIndex().
 *
 * Après réduction, il est possible récupérer la valeur réduite via
 * reducedValue().
 *
 * Les instances de cette classe peuvent être utilisées plusieurs fois.
 *
 * Voici un exemple pour calculer la somme d'un tableau de 50 éléments:
 *
 * \code
 * using namespace Arcane;
 * const Int32 nb_value(50);
 * Arcane::NumArray<Real, MDDim1> t1(nv_value);
 * Arcane::SmallSpan<const Real> t1_view(t1);
 * Arcane::RunQueue queue = ...;
 * Arcane::Accelerator::GenericReducer<Real> reducer(queue);
 *
 * // Calcul direct
 * reducer.applySum(t1_view);
 * std::cout << "Sum is '" << reducer.reducedValue() << "\n";
 *
 * // Calcul avec lambda
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
: private impl::GenericReducerBase<DataType>
{
 public:

  explicit GenericReducer(const RunQueue& queue)
  : impl::GenericReducerBase<DataType>(queue)
  {
    this->_allocate();
  }

 public:

  //! Applique une réduction 'Min' sur les valeurs \a values
  void applyMin(SmallSpan<const DataType> values)
  {
    _apply(values.size(), values.data(), impl::MinOperator<DataType>{});
  }

  //! Applique une réduction 'Max' sur les valeurs \a values
  void applyMax(SmallSpan<const DataType> values)
  {
    _apply(values.size(), values.data(), impl::MaxOperator<DataType>{});
  }

  //! Applique une réduction 'Somme' sur les valeurs \a values
  void applySum(SmallSpan<const DataType> values)
  {
    _apply(values.size(), values.data(), impl::SumOperator<DataType>{});
  }

  //! Applique une réduction 'Min' sur les valeurs sélectionnées par \a select_lambda
  template <typename SelectLambda>
  void applyMinWithIndex(Int32 nb_value, const SelectLambda& select_lambda)
  {
    _applyWithIndex(nb_value, select_lambda, impl::MinOperator<DataType>{});
  }

  //! Applique une réduction 'Max' sur les valeurs sélectionnées par \a select_lambda
  template <typename SelectLambda>
  void applyMaxWithIndex(Int32 nb_value, const SelectLambda& select_lambda)
  {
    _applyWithIndex(nb_value, select_lambda, impl::MaxOperator<DataType>{});
  }

  //! Applique une réduction 'Somme' sur les valeurs sélectionnées par \a select_lambda
  template <typename SelectLambda>
  void applySumWithIndex(Int32 nb_value, const SelectLambda& select_lambda)
  {
    _applyWithIndex(nb_value, select_lambda, impl::SumOperator<DataType>{});
  }

  //! Valeur de la réduction
  DataType reducedValue()
  {
    m_is_already_called = false;
    return this->_reducedValue();
  }

 private:

  bool m_is_already_called = false;

 private:

  template <typename InputIterator, typename ReduceOperator>
  void _apply(Int32 nb_value, InputIterator input_iter, ReduceOperator reduce_op)
  {
    _setCalled();
    impl::GenericReducerBase<DataType>* base_ptr = this;
    impl::GenericReducerIf<DataType> gf;
    DataType init_value = reduce_op.defaultValue();
    gf.apply(*base_ptr, nb_value, init_value, input_iter, reduce_op);
  }

  template <typename GetterLambda, typename ReduceOperator>
  void _applyWithIndex(Int32 nb_value, const GetterLambda& getter_lambda, ReduceOperator reduce_op)
  {
    _setCalled();
    impl::GenericReducerBase<DataType>* base_ptr = this;
    impl::GenericReducerIf<DataType> gf;
    impl::GetterLambdaIterator<DataType, GetterLambda> input_iter(getter_lambda);
    DataType init_value = reduce_op.defaultValue();
    gf.apply(*base_ptr, nb_value, init_value, input_iter, reduce_op);
  }

  void _setCalled()
  {
    if (m_is_already_called)
      ARCANE_FATAL("apply() has already been called for this instance");
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
