// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobal.h                                     (C) 2000-2024 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_ACCELERATORCOREGLOBAL_H
#define ARCANE_ACCELERATOR_CORE_ACCELERATORCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*!
 * \file AcceleratorCoreGlobal.h
 *
 * Ce fichier contient les déclarations des types de la composante
 * 'arcane_accelerator_core'.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_accelerator_core
#define ARCANE_ACCELERATOR_CORE_EXPORT ARCANE_EXPORT
#else
#define ARCANE_ACCELERATOR_CORE_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAcceleratorMng;
class Runner;
class RunQueue;
class RunQueuePool;
class RunCommand;
class RunQueueEvent;
class AcceleratorRuntimeInitialisationInfo;
class RunQueueBuildInfo;
class MemoryCopyArgs;
class MemoryPrefetchArgs;
class DeviceId;
class DeviceInfo;
class IDeviceInfoList;
class PointerAttribute;
class ViewBuildInfo;
enum class eMemoryAdvice;

namespace impl
{
  class RuntimeStaticInfo;
  class IRunnerRuntime;
  // typedef pour compatibilité avec anciennes versions (octobre 2022)
  using IRunQueueRuntime = IRunnerRuntime;
  class IRunQueueStream;
  class RunCommandImpl;
  class IReduceMemoryImpl;
  class ReduceMemoryImpl;
  class RunQueueImpl;
  class IRunQueueEventImpl;
  class RunCommandLaunchInfo;
  class RunnerImpl;
  class RunQueueImplStack;
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique d'exécution pour un Runner.
 */
enum class eExecutionPolicy
{
  //! Aucune politique d'exécution
  None,
  //! Politique d'exécution séquentielle
  Sequential,
  //! Politique d'exécution multi-thread
  Thread,
  //! Politique d'exécution utilisant l'environnement CUDA
  CUDA,
  //! Politique d'exécution utilisant l'environnement HIP
  HIP,
  //! Politique d'exécution utilisant l'environnement SYCL
  SYCL
};

//! Affiche le nom de la politique d'exécution
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
std::ostream&
operator<<(std::ostream& o, eExecutionPolicy exec_policy);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique des opératations de réduction sur les accélérateurs
 */
enum class eDeviceReducePolicy
{
  //! Utilise des opérations atomiques entre les blocs
  Atomic = 1,
  //! Utilise un noyau de calcul avec une synchronisations entre les blocs.
  Grid = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Niveaux de priorité prédéfinis pour les files d'exécution 
 *        sur les accélérateurs
 */
enum class eRunQueuePriority : int
{
  //! Utilise 0 comme valeur par défaut
  Default = 0,
  //! Une valeur arbitraire négative pour définir une priorité élevée
  High = -100,
  //! Une valeur arbitraire positive pour définir une priorité faible
  Low = 100
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type de mémoire pour un pointeur
enum class ePointerMemoryType
{
  //NOTE: Les valeurs sont équivalentes à cudaMemoryType. Si on
  // change ces valeurs il faut changer la fonction correspondante
  // dans le runtime (getPointerAttribute()).
  Unregistered = 0,
  Host = 1,
  Device = 2,
  Managed = 3
};

//! Affiche le nom du type de mémoire
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
std::ostream&
operator<<(std::ostream& o, ePointerMemoryType mem_type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations d'accessibilité d'une adresse mémoire.
 *
 * Indique si une adresse mémoire est accessible sur un accélérateur ou
 * sur le CPU.
 *
 * \sa getPointerAccessibility()
 */
enum class ePointerAccessibility
{
  //! Accessibilité inconnue
  Unknown = 0,
  //! Non accessible
  No = 1,
  //! Accessible
  Yes = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si \a exec_policy correspond à un accélérateur
inline bool
isAcceleratorPolicy(eExecutionPolicy exec_policy)
{
  return exec_policy == eExecutionPolicy::CUDA || exec_policy == eExecutionPolicy::HIP || exec_policy == eExecutionPolicy::SYCL;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Accessibilité de l'adresse \a ptr pour une exécution sur la file \a queue.
 *
 * Si \a queue est nul, retourne ePointerAccessibility::Unknown.
 * Si \a ptr_attr est non nul, il sera remplit avec les informations du pointeur
 * comme si on avait appelé Runner::fillPointerAttribute().
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT ePointerAccessibility
getPointerAccessibility(RunQueue* queue, const void* ptr, PointerAttribute* ptr_attr = nullptr);

/*!
 * \brief Accessibilité de l'adresse \a ptr pour une exécution sur \a runner.
 *
 * Si \a runner est nul, retourne ePointerAccessibility::Unknown.
 * Si \a ptr_attr est non nul, il sera remplit avec les informations du pointeur
 * comme si on avait appelé Runner::fillPointerAttribute().
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT ePointerAccessibility
getPointerAccessibility(Runner* runner, const void* ptr, PointerAttribute* ptr_attr = nullptr);

/*!
 * \brief Accessibilité de l'adresse \a ptr pour une politique d'exécution\a policy.
 *
 * Si \a ptr_attr est non nul, il sera remplit avec les informations du pointeur
 * comme si on avait appelé Runner::fillPointerAttribute().
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT ePointerAccessibility
getPointerAccessibility(eExecutionPolicy policy, const void* ptr, PointerAttribute* ptr_attr = nullptr);

//! Accessibilité de l'adresse \a ptr pour une exécution sur \a queue_or_runner_or_policy.
template <typename T> inline ePointerAccessibility
getPointerAccessibility(T& queue_or_runner_or_policy, const void* ptr, PointerAttribute* ptr_attr = nullptr)
{
  return getPointerAccessibility(&queue_or_runner_or_policy, ptr, ptr_attr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
using Arcane::Accelerator::isAcceleratorPolicy;

/*!
 * \brief Vérifie si \a ptr est accessible pour une exécution sur \a queue.
 *
 * Lève une exception FatalErrorException si ce n'est pas le cas.
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneCheckPointerIsAccessible(RunQueue* queue, const void* ptr,
                               const char* name, const TraceInfo& ti);

/*!
 * \brief Vérifie si \a ptr est accessible pour une exécution sur \a runner.
 *
 * Lève une exception FatalErrorException si ce n'est pas le cas.
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneCheckPointerIsAccessible(Runner* runner, const void* ptr,
                               const char* name, const TraceInfo& ti);

/*!
 * \brief Vérifie si \a ptr est accessible pour une exécution \a policy.
 *
 * Lève une exception FatalErrorException si ce n'est pas le cas.
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneCheckPointerIsAccessible(eExecutionPolicy policy, const void* ptr,
                               const char* name, const TraceInfo& ti);

template <typename T> inline void
arcaneCheckPointerIsAccessible(T& queue_or_runner, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  arcaneCheckPointerIsAccessible(&queue_or_runner, ptr, name, ti);
}

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro qui vérifie si \a ptr est accessible pour une RunQueue ou un Runner.
 *
 * Lance une exception si ce n'est pas le cas.
 */
#define ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue_or_runner_or_policy, ptr) \
  ::Arcane::Accelerator::impl::arcaneCheckPointerIsAccessible((queue_or_runner_or_policy), (ptr), #ptr, A_FUNCINFO)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK
//! Macro qui vérifie en mode check si \a ptr est accessible pour une RunQueue ou un Runner.
#define ARCANE_CHECK_ACCESSIBLE_POINTER(queue_or_runner_or_policy, ptr)      \
  ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS((queue_or_runner_or_policy), (ptr))
#else
//! Macro qui vérifie en mode check si \a ptr est accessible pour une RunQueue ou un Runner.
#define ARCANE_CHECK_ACCESSIBLE_POINTER(queue_or_runner_or_policy, ptr)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
