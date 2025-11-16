// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunch.h                                          (C) 2000-2025 */
/*                                                                           */
/* RunCommand pour le parallélisme hiérarchique.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDLAUNCH_H
#define ARCANE_ACCELERATOR_RUNCOMMANDLAUNCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/WorkGroupLoopRange.h"

#include "arcane/accelerator/RunCommandLoop.h"

#include "arcane/core/Concurrency.h"

#include "arccore/common/SequentialFor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé un intervalle d'itération pour la commande \a command pour \a nb_group de taille \a block_size
extern "C++" ARCANE_ACCELERATOR_EXPORT WorkGroupLoopRange
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 block_size);

//! Créé un intervalle d'itération pour la commande \a command pour \a nb_element
extern "C++" ARCANE_ACCELERATOR_EXPORT WorkGroupLoopRange
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_element);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

//! Classe pour exécuter en séquentiel sur l'hôte une partie de la boucle.
class WorkGroupSequentialForHelper
{
 public:

  //! Applique le fonctor \a func sur une boucle séqentielle.
  template <typename Lambda, typename... RemainingArgs> static void
  apply(Int32 begin_index, Int32 nb_loop, WorkGroupLoopRange bounds,
        const Lambda& func, RemainingArgs... remaining_args)
  {
    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtBegin(remaining_args...);
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

    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtEnd(remaining_args...);
  }
};

} // namespace Arcane::Accelerator::Impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique le fonctor \a func sur une boucle séqentielle.
template <typename Lambda, typename... RemainingArgs> void
arcaneSequentialFor(WorkGroupLoopRange bounds, const Lambda& func, const RemainingArgs&... remaining_args)
{
  Impl::WorkGroupSequentialForHelper::apply(0, bounds.nbGroup(), bounds, func, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique le fonctor \a func sur une boucle parallèle
template <typename Lambda, typename... RemainingArgs> void
arccoreParallelFor(WorkGroupLoopRange bounds, ForLoopRunInfo run_info,
                   const Lambda& func, const RemainingArgs&... remaining_args)
{
  auto sub_func = [=](Int32 begin_index, Int32 nb_loop) {
    Impl::WorkGroupSequentialForHelper::apply(begin_index, nb_loop, bounds, func, remaining_args...);
  };
  arcaneParallelFor(0, bounds.nbGroup(), run_info, sub_func);
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
