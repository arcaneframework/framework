// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchImpl.h                                      (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'une RunCommand pour le parallélisme hiérarchique.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDLAUNCHIMPL_H
#define ARCANE_ACCELERATOR_RUNCOMMANDLAUNCHIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/WorkGroupLoopRange.h"

#include "arccore/accelerator/RunCommandLoop.h"

//#include "arcane/core/Concurrency.h"

#include "arccore/common/SequentialFor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*!
 * \internal
 * \brief Classe pour exécuter en séquentiel sur l'hôte une partie de la boucle.
 */
class WorkGroupSequentialForHelper
{
 public:

  //! Applique le fonctor \a func sur une boucle séqentielle.
  template <typename Lambda, typename... RemainingArgs> static void
  apply(Int32 begin_index, Int32 nb_loop, WorkGroupLoopRange bounds,
        const Lambda& func, RemainingArgs... remaining_args)
  {
    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtBegin(remaining_args...);
    const Int32 group_size = bounds.groupSize();
    Int32 loop_index = begin_index * group_size;
    for (Int32 i = begin_index; i < (begin_index + nb_loop); ++i) {
      // Pour la dernière itération de la boucle, le nombre d'éléments actifs peut-être
      // inférieur à la taille d'un groupe si \a total_nb_element n'est pas
      // un multiple de \a group_size.
      Int32 nb_active = bounds.nbActiveItem(i);
      func(WorkGroupLoopContext(loop_index, i, group_size, nb_active), remaining_args...);
      loop_index += group_size;
    }

    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtEnd(remaining_args...);
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
 * \internal
 * \brief Applique le fonctor \a func sur une boucle séqentielle.
 */
template <typename Lambda, typename... RemainingArgs> void
arccoreSequentialFor(WorkGroupLoopRange bounds, const Lambda& func, const RemainingArgs&... remaining_args)
{
  Impl::WorkGroupSequentialForHelper::apply(0, bounds.nbGroup(), bounds, func, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Applique le fonctor \a func sur une boucle parallèle.
 */
template <typename Lambda, typename... RemainingArgs> void
arccoreParallelFor(WorkGroupLoopRange bounds, ForLoopRunInfo run_info,
                   const Lambda& func, const RemainingArgs&... remaining_args)
{
  auto sub_func = [=](Int32 begin_index, Int32 nb_loop) {
    Impl::WorkGroupSequentialForHelper::apply(begin_index, nb_loop, bounds, func, remaining_args...);
  };
  arccoreParallelFor(0, bounds.nbGroup(), run_info, sub_func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
