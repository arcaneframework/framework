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
#ifndef ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCH_H
#define ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/RunCommandLaunchImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un intervalle d'itération pour la commande \a command.
 *
 * Créé un intervalle pour \a nb_group de taille \a group_size.
 * Le nombre total d'éléments est donc égal à `nb_group * group_size`.
 */
extern "C++" ARCCORE_ACCELERATOR_EXPORT WorkGroupLoopRange
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un intervalle d'itération pour la commande \a command.
 *
 * Créé un intervalle contenant \a nb_element, répartis en \a nb_group de taille \a group_size.
 * Si \a nb_group et \a group_size sont nuls, une taille de bloc par défaut sera choisie en
 * fonction de l'accélérateur et \a nb_group sera calculé automatiquement.
 */
extern "C++" ARCCORE_ACCELERATOR_EXPORT WorkGroupLoopRange
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_element, Int32 nb_group, Int32 group_size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour Sycl, le type de l'itérateur ne peut pas être le même sur l'hôte et
// le device car il faut un 'sycl::nd_item' et il n'est pas possible d'en
// construire un (pas de constructeur par défaut). On utilise donc
// une lambda template et le type de l'itérateur est un paramètre template

/*!
 * \brief Macro pour lancer une commande utilisant le parallélisme hiérarchique.
 *
 * \a bounds doit être une instance de Arcane::Accelerator::WorkGroupLoopRange.
 * La création de ces instances se fait via l'appel à
 * Arcane::Accelerator::makeWorkGroupLoopRange().
 *
 * \a iter_name sera du type Arcane::Accelerator::WorkGroupLoopContext sauf
 * pour la politique d'exécution Arcane::Accelerator::eExecutionPolicy::SYCL
 * où il sera du type Arcane::Accelerator::SyclWorkGroupLoopContext.
 */
#if defined(ARCCORE_COMPILING_SYCL)
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(auto iter_name __VA_OPT__(ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))
#else
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(typename decltype(bounds)::LoopIndexType iter_name __VA_OPT__(ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
