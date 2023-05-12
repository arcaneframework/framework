// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobal.h                                     (C) 2000-2022 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_ACCELERATORCOREGLOBAL_H
#define ARCANE_ACCELERATOR_CORE_ACCELERATORCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

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
  HIP
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
  return exec_policy == eExecutionPolicy::CUDA || exec_policy == eExecutionPolicy::HIP;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Accessibilité de l'adresse \a ptr pour une exécution sur la file \a queue.
 *
 * Si \a queue est nul, retourne ePointerAccessibility::Unknown.
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT ePointerAccessibility
getPointerAccessibility(RunQueue* queue, const void* ptr);

//! Accessibilité de l'adresse \a ptr pour une exécution sur \a queue.
inline ePointerAccessibility
getPointerAccessibility(RunQueue& queue, const void* ptr)
{
  return getPointerAccessibility(&queue, ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Accessibilité de l'adresse \a ptr pour une exécution sur \a runner.
 *
 * Si \a runner est nul, retourne ePointerAccessibility::Unknown.
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT ePointerAccessibility
getPointerAccessibility(Runner* runner, const void* ptr);

//! Accessibilité de l'adresse \a ptr pour une exécution sur \a runner.
inline ePointerAccessibility
getPointerAccessibility(Runner& runner, const void* ptr)
{
  return getPointerAccessibility(&runner, ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
using Arcane::Accelerator::isAcceleratorPolicy;

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneThrowPointerNotAcccessible [[noreturn]] (const void* ptr, const TraceInfo& ti);

inline void
arcaneCheckPointerIsAcccessible(ePointerAccessibility a, const void* ptr, const TraceInfo& ti)
{
  if (a == ePointerAccessibility::No)
    arcaneThrowPointerNotAcccessible(ptr, ti);
}

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro qui vérifie si \a ptr est accessible pour une RunQueue ou un Runner.
 *
 * Lance une exception si ce n'est pas le cas.
 */
#define ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue_or_runner, ptr) \
  ::Arcane::Accelerator::impl::arcaneCheckPointerIsAcccessible(::Arcane::Accelerator::getPointerAccessibility((queue_or_runner), (ptr)), (ptr), A_FUNCINFO)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK
//! Macro qui vérifie en mode check si \a ptr est accessible pour une RunQueue ou un Runner.
#define ARCANE_CHECK_ACCESSIBLE_POINTER(queue_or_runner, ptr) \
  ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS((queue_or_runner), (ptr))
#else
//! Macro qui vérifie en mode check si \a ptr est accessible pour une RunQueue ou un Runner.
#define ARCANE_CHECK_ACCESSIBLE_POINTER(queue_or_runner, ptr)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
