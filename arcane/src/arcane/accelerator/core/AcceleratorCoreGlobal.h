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
}

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
std::ostream& operator<<(std::ostream& o,eExecutionPolicy exec_policy);

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

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si \a exec_policy correspond à un accélérateur
inline bool
isAcceleratorPolicy(eExecutionPolicy exec_policy)
{
  return exec_policy==eExecutionPolicy::CUDA || exec_policy==eExecutionPolicy::HIP;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
