// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorGlobal.h                                         (C) 2000-2020 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_ACCELERATORGLOBAL_H
#define ARCANE_ACCELERATOR_ACCELERATORGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_accelerator
#define ARCANE_ACCELERATOR_EXPORT ARCANE_EXPORT
#else
#define ARCANE_ACCELERATOR_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Runner;
class RunQueue;
class RunCommand;
class IRunQueueRuntime;
namespace impl
{
class IReduceMemoryImpl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique d'exécution.
 */
enum class eExecutionPolicy
{
  Sequential,
  Thread,
  CUDA,
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si \a exec_policy correspond à un accélérateur
inline bool
isAcceleratorPolicy(eExecutionPolicy exec_policy)
{
  return exec_policy==eExecutionPolicy::CUDA;
}

//! Indique si on utilise le runtime CUDA
extern "C++" ARCANE_ACCELERATOR_EXPORT
bool isUsingCUDARuntime();

//! Positionne l'utilisation du runtime CUDA
extern "C++" ARCANE_ACCELERATOR_EXPORT
void setUsingCUDARuntime(bool v);

//! Récupère l'implémentation CUDA de RunQueue (peut être nulle)
extern "C++" ARCANE_ACCELERATOR_EXPORT
IRunQueueRuntime* getCUDARunQueueRuntime();

//! Positionne l'implémentation CUDA de RunQueue.
extern "C++" ARCANE_ACCELERATOR_EXPORT
void setCUDARunQueueRuntime(IRunQueueRuntime* v);

//! Récupère l'implémentation Séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_EXPORT
IRunQueueRuntime* getSequentialRunQueueRuntime();

namespace impl
{
extern "C++" ARCANE_ACCELERATOR_EXPORT IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
