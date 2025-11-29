// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobal.h                                     (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_COMMONACCELERATORCOREGLOBAL_H
#define ARCCORE_COMMON_ACCELERATOR_COMMONACCELERATORCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"
#include "arccore/trace/TraceGlobal.h"

#include <iosfwd>

/*!
 * \file AcceleratorCoreGlobal.h
 *
 * Ce fichier contient les déclarations des types de la composante
 * 'arcane_accelerator_core'.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
class DeviceMemoryInfo;
class ProfileRegion;
class IDeviceInfoList;
class PointerAttribute;
class ViewBuildInfo;
class RunnerInternal;
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
  class RunnerImpl;
  class RunQueueImplStack;
} // namespace impl

namespace Impl
{
  class KernelLaunchArgs;
  class RunCommandLaunchInfo;
  class NativeStream;
  class CudaUtils;
  class HipUtils;
  class SyclUtils;
} // namespace Impl

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
extern "C++" ARCCORE_COMMON_EXPORT
std::ostream&
operator<<(std::ostream& o, eExecutionPolicy exec_policy);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique des opératations de réduction sur les accélérateurs.
 *
 * \note A partir de la version 3.15 de Arcane, seule la politique Grid
 * est disponible.
 */
enum class eDeviceReducePolicy
{
  /*!
   * \brief Utilise des opérations atomiques entre les blocs.
   *
   * \deprecated Cette politique n'est plus disponible. Si on
   * spécifie cette politique, elle se comportera comme
   * eDeviceReducePolicy::Grid.
   */
  Atomic ARCCORE_DEPRECATED_REASON("Y2025: Use eDeviceReducePolicy::Grid instead") = 1,
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
extern "C++" ARCCORE_COMMON_EXPORT
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
extern "C++" ARCCORE_COMMON_EXPORT ePointerAccessibility
getPointerAccessibility(RunQueue* queue, const void* ptr, PointerAttribute* ptr_attr = nullptr);

/*!
 * \brief Accessibilité de l'adresse \a ptr pour une exécution sur \a runner.
 *
 * Si \a runner est nul, retourne ePointerAccessibility::Unknown.
 * Si \a ptr_attr est non nul, il sera remplit avec les informations du pointeur
 * comme si on avait appelé Runner::fillPointerAttribute().
 */
extern "C++" ARCCORE_COMMON_EXPORT ePointerAccessibility
getPointerAccessibility(Runner* runner, const void* ptr, PointerAttribute* ptr_attr = nullptr);

/*!
 * \brief Accessibilité de l'adresse \a ptr pour une politique d'exécution\a policy.
 *
 * Si \a ptr_attr est non nul, il sera remplit avec les informations du pointeur
 * comme si on avait appelé Runner::fillPointerAttribute().
 */
extern "C++" ARCCORE_COMMON_EXPORT ePointerAccessibility
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
extern "C++" ARCCORE_COMMON_EXPORT void
arcaneCheckPointerIsAccessible(const RunQueue* queue, const void* ptr,
                               const char* name, const TraceInfo& ti);

/*!
 * \brief Vérifie si \a ptr est accessible pour une exécution sur \a runner.
 *
 * Lève une exception FatalErrorException si ce n'est pas le cas.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
arcaneCheckPointerIsAccessible(const Runner* runner, const void* ptr,
                               const char* name, const TraceInfo& ti);

/*!
 * \brief Vérifie si \a ptr est accessible pour une exécution \a policy.
 *
 * Lève une exception FatalErrorException si ce n'est pas le cas.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
arcaneCheckPointerIsAccessible(eExecutionPolicy policy, const void* ptr,
                               const char* name, const TraceInfo& ti);

inline void
arcaneCheckPointerIsAccessible(const RunQueue& queue, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  arcaneCheckPointerIsAccessible(&queue, ptr, name, ti);
}

inline void
arcaneCheckPointerIsAccessible(const Runner& runner, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  arcaneCheckPointerIsAccessible(&runner, ptr, name, ti);
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
