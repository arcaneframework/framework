// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WorkGroupLoopRange.h                                        (C) 2000-2025 */
/*                                                                           */
/* Boucle pour le parallélisme hiérarchique.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_WORKGROUPLOOPRANGE_H
#define ARCANE_ACCELERATOR_WORKGROUPLOOPRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arccore/common/SequentialFor.h"

#if defined(ARCANE_COMPILING_CUDA)
#include <cooperative_groups.h>
#endif
#if defined(ARCANE_COMPILING_HIP)
#include <hip/hip_cooperative_groups.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

class T0
{
 public:

  constexpr operator int() const { return 2; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index d'un WorkItem dans un WorkGroupLoopRange.
 */
class WorkGroupLoopIndex
{
 public:

  //! Constructeur pour le device (la taille du groupe est dans blockIdx.x)
  explicit constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex(Int32 index)
  : m_index(index)
  {}
  //! Constructeur pour l'hôte
  explicit constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex(Int32 index, Int32 group_index, Int32 group_size)
  : m_index(index)
  , m_group_size(group_size)
  , m_group_index(group_index)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 operator()() const { return m_index; }

  /*!
   * \brief Rang du groupe du WorkItem dans la liste des WorkGroup.
   */
  ARCCORE_HOST_DEVICE Int32 groupRank() const
  {
#if defined(ARCCORE_DEVICE_CODE)
    return blockIdx.x;
#else
    return m_group_index;
#endif
  }
  /*!
   * \brief Nombre de WorkItem dans un WorkGroup.
   */
  ARCCORE_HOST_DEVICE Int32 groupSize() const
  {
#if defined(ARCCORE_DEVICE_CODE)
    return blockDim.x;
#else
    return m_group_size;
#endif
  }
  /*!
   * \brief Rang du WorkItem dans son WorkGroup.
   */
  ARCCORE_HOST_DEVICE Int32 rankInGroup() const
  {
#if defined(ARCCORE_DEVICE_CODE)
    return threadIdx.x;
#else
    return m_index % m_group_size;
#endif
  }

  Int32 hostGroupIndex() const { return m_group_index; }
  Int32 hostGroupSize() const { return m_group_size; }

#if defined(ARCCORE_DEVICE_CODE)
  __device__ T0 x() const { return {}; }
#else
  int x() const { return 0; }
#endif

#if defined(ARCCORE_DEVICE_CODE)
  constexpr __device__ bool isDevice() const { return true; }
#else
  //TODO: gérer SYCL
  constexpr bool isDevice() const { return false; }
#endif

  void sync() {}

 private:

  Int32 m_index = 0;
  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
};

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index d'un WorkItem dans un WorkGroupLoopRange.
 */
class DeviceWorkItemBlock
{
 public:

  //! Constructeur pour le device (la taille du groupe est dans blockIdx.x)
  explicit __device__ DeviceWorkItemBlock(Int32 index)
  : m_index(index)
  , m_thread_block(cooperative_groups::this_thread_block())
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 operator()() const { return m_index; }

  /*!
   * \brief Rang du groupe du WorkItem dans la liste des WorkGroup.
   */
  __device__ Int32 groupRank() const { return m_thread_block.group_index().x; }

  /*!
   * \brief Nombre de WorkItem dans un WorkGroup.
   */
  __device__ Int32 groupSize() { return m_thread_block.group_dim().x; }

  /*!
   * \brief Rang du WorkItem dans son WorkGroup.
   */
  __device__ Int32 rankInGroup() const { return m_thread_block.thread_index().x; }

  __device__ void sync() { m_thread_block.sync(); }

  __device__ T0 x() const { return {}; }

  constexpr __device__ bool isDevice() const { return true; }

 private:

  Int32 m_index = 0;
  cooperative_groups::thread_block m_thread_block;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class WorkGroupLoopIndex2
{
 public:

  explicit constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex2(Int32 loop_index, Int32 group_index, Int32 group_size)
  : m_loop_index(loop_index)
  , m_group_index(group_index)
  , m_group_size(group_size)
  {}

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit ARCCORE_HOST_DEVICE WorkGroupLoopIndex2()
  {
#if defined(ARCCORE_DEVICE_CODE)
    m_loop_index = blockDim.x * blockIdx.x + threadIdx.x;
    m_group_index = blockIdx.x;
    m_group_size = blockDim.x;
#endif
  }

 public:

  constexpr Int32 nbItem()
  {
#if defined(ARCCORE_DEVICE_CODE)
    return 1;
#else
    return m_group_size;
#endif
  }

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCANE_COMPILING_SYCL)
  __device__ DeviceWorkItemBlock item([[maybe_unused]] Int32 index) const
  {
    // N'est valide que pour index==0
    return item0();
  }
#else
  WorkGroupLoopIndex item(Int32 index) const
  {
    return WorkGroupLoopIndex(m_loop_index + index, m_group_index, m_group_size);
  }
#endif

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCANE_COMPILING_SYCL)
  __device__ DeviceWorkItemBlock item0() const
  {
    // N'est valide que pour index==0
    return DeviceWorkItemBlock(blockDim.x * blockIdx.x + threadIdx.x);
  }
#else
  WorkGroupLoopIndex item0() const
  {
    return WorkGroupLoopIndex(m_loop_index, m_group_index, m_group_size);
  }
#endif

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_index = 0;
  Int32 m_group_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Intervalle d'itération d'une boucle utilisant le parallélisme hiérarchique.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors d'Arcane.
 *
 * L'intervalle d'itération est décomposé en \a N WorkGroup contenant chacun \a P WorkItem.
 */
class WorkGroupLoopRange
{
 public:

  //! Type de l'index de la boucle
  using LoopIndexType = WorkGroupLoopIndex2;

 public:

  explicit WorkGroupLoopRange(Int32 total_size)
  : m_total_size(total_size)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 totalSize() const { return m_total_size; }
  constexpr ARCCORE_HOST_DEVICE Int64 nbElement() const { return m_total_size; }
  constexpr ARCCORE_HOST_DEVICE Int32 groupSize() const { return m_group_size; }

  constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex2 getIndices(Int32 x) const
  {
    // TODO: supprimer la division
    return WorkGroupLoopIndex2(x, x / m_group_size, m_group_size);
  }

 private:

  Int32 m_total_size = 0;
  Int32 m_group_size = 256;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Applique le fonctor \a func sur une boucle séqentielle.
template <typename Lambda, typename... ReducerArgs>
inline void
arcaneSequentialFor(Impl::WorkGroupLoopRange bounds, const Lambda& func, ReducerArgs... reducer_args)
{
  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtBegin(reducer_args...);
  const Int32 group_size = bounds.groupSize();
  const Int32 total_size = bounds.totalSize();
  // TODO: gérer si total_size n'est pas un multiple de group_size
  const Int32 nb_group = total_size / group_size;
  Int32 loop_index = 0;
  for (Int32 i = 0; i < nb_group; ++i) {
    func(Impl::WorkGroupLoopIndex2(loop_index, i, group_size), reducer_args...);
    loop_index += group_size;
  }

  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtEnd(reducer_args...);
}

//! Applique le fonctor \a func sur une boucle parallèle
template <typename Lambda, typename... ReducerArgs>
inline void
arccoreParallelFor(Impl::WorkGroupLoopRange bounds,
                   [[maybe_unused]] const ForLoopRunInfo& run_info,
                   const Lambda& func, ReducerArgs... reducer_args)
{
  // Pour l'instant on ne fait que du séquentiel.
  arcaneSequentialFor(bounds, func, reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
