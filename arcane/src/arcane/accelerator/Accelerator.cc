// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Accelerator.cc                                              (C) 2000-2021 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/Reduce.h"

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
IRunQueueRuntime* global_cuda_runqueue_runtime = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_EXPORT
bool isUsingCUDARuntime()
{
  return global_is_using_cuda_runtime;
}

extern "C++" ARCANE_ACCELERATOR_EXPORT
void setUsingCUDARuntime(bool v)
{
  global_is_using_cuda_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation CUDA de RunQueue
extern "C++" ARCANE_ACCELERATOR_EXPORT
IRunQueueRuntime* getCUDARunQueueRuntime()
{
  return global_cuda_runqueue_runtime;
}

//! Positionne l'implémentation CUDA de RunQueue.
extern "C++" ARCANE_ACCELERATOR_EXPORT
void setCUDARunQueueRuntime(IRunQueueRuntime* v)
{
  global_cuda_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Affiche le nom de la politique d'exécution
extern "C++" ARCANE_ACCELERATOR_EXPORT
ostream& operator<<(ostream& o,eExecutionPolicy exec_policy)
{
  switch(exec_policy){
  case eExecutionPolicy::Sequential: o << "Sequential"; break;
  case eExecutionPolicy::Thread: o << "Thread"; break;
  case eExecutionPolicy::CUDA: o << "CUDA"; break;
  }
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
