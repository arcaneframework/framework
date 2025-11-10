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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

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
  explicit constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex(Int32 index, Int32 group_size)
  : m_index(index)
  , m_group_size(group_size)
  , m_group_index(index / group_size)
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
  /*!
   * \brief Barrière sur tous les WorkItem du WorkGroup.
   */
  ARCCORE_HOST_DEVICE void barrier() const
  {
#if defined(ARCCORE_DEVICE_CODE)
    __syncthreads();
#endif
  }

  Int32 hostGroupIndex() const { return m_group_index; }
  Int32 hostGroupSize() const { return m_group_size; }

 private:

  Int32 m_index = 0;
  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
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
  using LoopIndexType = WorkGroupLoopIndex;

 public:

  explicit WorkGroupLoopRange(Int32 total_size)
  : m_total_size(total_size)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 totalSize() const { return m_total_size; }
  constexpr ARCCORE_HOST_DEVICE Int64 nbElement() const { return m_total_size; }
  constexpr ARCCORE_HOST_DEVICE Int32 groupSize() const { return m_group_size; }
  constexpr ARCCORE_HOST_DEVICE WorkGroupLoopIndex getIndices(Int32 x) const { return WorkGroupLoopIndex(x); }

 private:

  Int32 m_total_size = 0;
  Int32 m_group_size = 256;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class WorkGroupRemainingArgsHandler
{
 public:

  //! Applique les fonctors des arguments additionnels au début de l'itération
  template <typename... ReducerArgs> static inline void
  applyRemainingArgsAtBegin(ReducerArgs&... reducer_args)
  {
    (WorkGroupRemainingArgsHandler::_doOneAtBegin(reducer_args), ...);
  }

  //! Applique les fonctors des arguments additionnels à la fin de l'itération
  template <typename... ReducerArgs> static inline void
  applyRemainingArgsAtEnd(ReducerArgs&... reducer_args)
  {
    (WorkGroupRemainingArgsHandler::_doOneAtEnd(reducer_args), ...);
  }

 private:

  template <typename OneArg> static inline void _doOneAtBegin(OneArg& one_arg)
  {
    if constexpr (requires { one_arg._internalHostExecWorkItemAtBegin(); })
      one_arg._internalHostExecWorkItemAtBegin();
  }
  template <typename OneArg> static inline void _doOneAtEnd(OneArg& one_arg)
  {
    if constexpr (requires { one_arg._internalHostExecWorkItemAtEnd(); })
      one_arg._internalHostExecWorkItemAtEnd();
  }
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
  Impl::WorkGroupRemainingArgsHandler::applyRemainingArgsAtBegin(reducer_args...);
  Int32 group_size = bounds.groupSize();
  for (Int32 i0 = 0, n = bounds.totalSize(); i0 < n; ++i0)
    func(Impl::WorkGroupLoopIndex(i0, group_size), reducer_args...);
  Impl::WorkGroupRemainingArgsHandler::applyRemainingArgsAtEnd(reducer_args...);
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
