// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunQueueRuntime.h                                          (C) 2000-2021 */
/*                                                                           */
/* Implémentation d'un RunQueue pour une cible donnée.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_IRUNQUEUERUNTIME_H
#define ARCANE_ACCELERATOR_IRUNQUEUERUNTIME_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un runtime d'exécution sur calculateur.
 */
class ARCANE_ACCELERATOR_EXPORT IRunQueueRuntime
{
 public:
  virtual ~IRunQueueRuntime() = default;
 public:
  virtual void notifyBeginKernel() =0;
  virtual void notifyEndKernel() =0;
  virtual void barrier() =0;
  virtual eExecutionPolicy executionPolicy() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
