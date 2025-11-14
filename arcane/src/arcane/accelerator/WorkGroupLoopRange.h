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
#include "arcane/accelerator/RunCommandLoop.h"

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
class WorkGroupLoopRange;
class WorkGroupLoopIndex;

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
class HostWorkItemBlock
{
  //friend WorkGroupLoopRange;
  friend WorkGroupLoopIndex;

 private:

  //! Constructeur pour l'hôte
  explicit constexpr ARCCORE_HOST_DEVICE HostWorkItemBlock(Int32 index, Int32 group_index, Int32 group_size)
  : m_index(index)
  , m_group_size(group_size)
  , m_group_index(group_index)
  {}

 public:

  constexpr Int32 operator()() const { return m_index; }

  /*!
   * \brief Rang du groupe du WorkItem dans la liste des WorkGroup.
   */
  constexpr Int32 groupRank() const { return m_group_index; }
  /*!
   * \brief Nombre de WorkItem dans un WorkGroup.
   */
  constexpr Int32 groupSize() const { return m_group_size; }
  /*!
   * \brief Rang du WorkItem dans son WorkGroup.
   */
  constexpr Int32 rankInGroup() const { return m_index % m_group_size; }

  int x() const { return 0; }

  constexpr bool isDevice() const { return false; }

  void sync() {}

 private:

  Int32 m_index = 0;
  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index d'un WorkItem dans un WorkGroupLoopRange pour un device CUDA ou ROCM.
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

class WorkGroupLoopIndex
{
  friend WorkGroupLoopRange;
  template <typename Lambda, typename... RemainingArgs>
  friend void arcaneSequentialFor(WorkGroupLoopRange bounds, const Lambda& func, RemainingArgs... remaining_args);

 private:

  explicit constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex(Int32 loop_index, Int32 group_index, Int32 group_size)
  : m_loop_index(loop_index)
  , m_group_index(group_index)
  , m_group_size(group_size)
  {}

  // Ce constructeur n'est utilisé que sur le device
  explicit ARCCORE_HOST_DEVICE WorkGroupLoopIndex()
  {
#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCANE_COMPILING_SYCL)
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
  HostWorkItemBlock item(Int32 index) const
  {
    return HostWorkItemBlock(m_loop_index + index, m_group_index, m_group_size);
  }
#endif

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCANE_COMPILING_SYCL)
  __device__ DeviceWorkItemBlock item0() const
  {
    // N'est valide que pour index==0
    return DeviceWorkItemBlock(blockDim.x * blockIdx.x + threadIdx.x);
  }
#else
  HostWorkItemBlock item0() const
  {
    return HostWorkItemBlock(m_loop_index, m_group_index, m_group_size);
  }
#endif

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_index = 0;
  Int32 m_group_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Implémentation pour SYCL.
 *
 * L'équivalent de \a cooperative_groups::thread_group() avec SYCL
 * est le \a sycl::nd_item<1>. Il est plus compliqué à utiliser pour deux
 * raisons:
 *
 * - il n'y a pas dans SYCL un équivalent de
 * \a cooperative_groups::this_thread_block(). Il faut utiliser la valeur
 * de \a sycl::nb_item<1> passé en argument du noyau de calcul.
 * - il n'y a pas de constructeurs par défaut pour \a sycl::nb_item<1>.
 *
 * Pour contourner ces deux problèmes, on utilise un type spécifique pour
 * gérer les noyaux en SYCL. Heureusement, il est possible d'utiliser les
 * lambda template avec SYCL. On utilise donc deux types pour gérer
 * les noyaux selon qu'on s'exécute sur le device SYCL ou sur l'hôte.
 *
 * TODO: regarder si avec la macro SYCL_DEVICE_ONLY il n'est pas possible
 * d'avoir le même type comportant des champs différents
 */
#if defined(ARCANE_COMPILING_SYCL)

/*!
 * \brief Index d'un WorkItem dans un WorkGroupLoopRange.
 */
class SyclDeviceWorkItemBlock
{
 public:

  explicit SyclDeviceWorkItemBlock(sycl::nd_item<1> n)
  : m_index(static_cast<Int32>(n.get_group(0) * n.get_local_range(0) + n.get_local_id(0)))
  , m_thread_block(n)
  {
  }

 public:

  constexpr Int32 operator()() const { return m_index; }

  /*!
   * \brief Rang du groupe du WorkItem dans la liste des WorkGroup.
   */
  Int32 groupRank() const { return static_cast<Int32>(m_thread_block.get_group(0)); }

  /*!
   * \brief Nombre de WorkItem dans un WorkGroup.
   */
  Int32 groupSize() { return static_cast<Int32>(m_thread_block.get_local_range(0)); }

  /*!
   * \brief Rang du WorkItem dans son WorkGroup.
   */
  Int32 rankInGroup() const { return static_cast<Int32>(m_thread_block.get_local_id(0)); }

  void sync() { m_thread_block.barrier(); }

  T0 x() const { return {}; }

  constexpr bool isDevice() const { return true; }

 private:

  Int32 m_index = 0;
  sycl::nd_item<1> m_thread_block;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index dans un WorkGroup pour le back-end Sycl.
 */
class SyclWorkGroupLoopIndex
{
  friend WorkGroupLoopRange;

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclWorkGroupLoopIndex(sycl::nd_item<1> n)
  : m_nd_item(n)
  {
  }

 public:

  constexpr Int32 nbItem() { return 1; }

  SyclDeviceWorkItemBlock item([[maybe_unused]] Int32 index) const
  {
    // N'est valide que pour index==0
    return item0();
  }

  SyclDeviceWorkItemBlock item0() const
  {
    // N'est valide que pour index==0
    return SyclDeviceWorkItemBlock(m_nd_item);
  }

 private:

  sycl::nd_item<1> m_nd_item;
};

#endif // ARCANE_COMPILING_SYCL

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
  using LoopIndexType = WorkGroupLoopIndex;

 public:

  explicit WorkGroupLoopRange(Int32 total_size)
  : m_total_size(total_size)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 totalSize() const { return m_total_size; }
  constexpr ARCCORE_HOST_DEVICE Int64 nbElement() const { return m_total_size; }
  constexpr ARCCORE_HOST_DEVICE Int32 groupSize() const { return m_group_size; }

  constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex getIndices(Int32 x) const
  {
    // TODO: supprimer la division
    return WorkGroupLoopIndex(x, x / m_group_size, m_group_size);
  }

#if defined(ARCANE_COMPILING_SYCL)
  SyclWorkGroupLoopIndex getIndices(sycl::nd_item<1> id) const
  {
    return SyclWorkGroupLoopIndex(id);
  }
#endif

 private:

  Int32 m_total_size = 0;
  Int32 m_group_size = 256;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Applique le fonctor \a func sur une boucle séqentielle.
template <typename Lambda, typename... RemainingArgs>
void arcaneSequentialFor(WorkGroupLoopRange bounds, const Lambda& func, RemainingArgs... remaining_args)
{
  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtBegin(remaining_args...);
  const Int32 group_size = bounds.groupSize();
  const Int32 total_size = bounds.totalSize();
  // TODO: gérer si total_size n'est pas un multiple de group_size
  const Int32 nb_group = total_size / group_size;
  Int32 loop_index = 0;
  for (Int32 i = 0; i < nb_group; ++i) {
    func(WorkGroupLoopIndex(loop_index, i, group_size), remaining_args...);
    loop_index += group_size;
  }

  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtEnd(remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

namespace Arcane::Accelerator
{

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

// Pour Sycl, le type de l'itérateur ne peut pas être le même sur l'hôte et
// le device car il faut un 'sycl::nd_item' et il n'est pas possible d'en
// construire un (pas de constructeur par défaut). On utilise donc
// une lambda template et le type de l'itérateur est un paramètre template
#if defined(ARCANE_COMPILING_SYCL)
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(auto iter_name __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))
#else
//! Macro pour lancer une commande avec le support du parallélisme hiérarchique
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(typename decltype(bounds)::LoopIndexType iter_name __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
