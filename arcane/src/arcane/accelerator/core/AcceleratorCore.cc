// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCore.cc                                          (C) 2000-2021 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \namespace Arcane::Accelerator
 *
 * \brief Espace de nom pour l'utilisation des accélérateurs.
 *
 * Toutes les classes et types utilisés pour la gestion des accélérateurs
 * sont dans ce namespace.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
bool global_is_using_cuda_runtime = false;
impl::IRunQueueRuntime* global_cuda_runqueue_runtime = nullptr;
bool global_is_using_hip_runtime = false;
impl::IRunQueueRuntime* global_hip_runqueue_runtime = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool impl::
isUsingCUDARuntime()
{
  return global_is_using_cuda_runtime;
}

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setUsingCUDARuntime(bool v)
{
  global_is_using_cuda_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation CUDA de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT impl::IRunQueueRuntime* impl::
getCUDARunQueueRuntime()
{
  return global_cuda_runqueue_runtime;
}

//! Positionne l'implémentation CUDA de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setCUDARunQueueRuntime(IRunQueueRuntime* v)
{
  global_cuda_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool impl::
isUsingHIPRuntime()
{
  return global_is_using_hip_runtime;
}

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setUsingHIPRuntime(bool v)
{
  global_is_using_hip_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation HIP de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT impl::IRunQueueRuntime* impl::
getHIPRunQueueRuntime()
{
  return global_hip_runqueue_runtime;
}

//! Positionne l'implémentation HIP de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setHIPRunQueueRuntime(impl::IRunQueueRuntime* v)
{
  global_hip_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Affiche le nom de la politique d'exécution
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
std::ostream& operator<<(std::ostream& o,eExecutionPolicy exec_policy)
{
  switch(exec_policy){
  case eExecutionPolicy::None: o << "None"; break;
  case eExecutionPolicy::Sequential: o << "Sequential"; break;
  case eExecutionPolicy::Thread: o << "Thread"; break;
  case eExecutionPolicy::CUDA: o << "CUDA"; break;
  case eExecutionPolicy::HIP: o << "HIP"; break;
  }
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
