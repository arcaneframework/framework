// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectExecutionFunctor.h                                    (C) 2000-2022 */
/*                                                                           */
/* Fonctor pour l'exécution directe.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_INTERNAL_DIRECTEXECUTIONFUNCTOR_H
#define ARCANE_LAUNCHER_INTERNAL_DIRECTEXECUTIONFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_LAUNCHER_EXPORT IDirectExecutionFunctor
{
 public:
  virtual ~IDirectExecutionFunctor(){}
  virtual int execute(DirectExecutionContext*)
  {
    return (-1);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_LAUNCHER_EXPORT IDirectSubDomainExecutionFunctor
{
 public:
  virtual ~IDirectSubDomainExecutionFunctor(){}
  virtual int execute(DirectSubDomainExecutionContext*)
  {
    return (-1);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_LAUNCHER_EXPORT DirectExecutionWrapper
{
 public:
  static int run(IDirectExecutionFunctor*);
  static int run(IDirectSubDomainExecutionFunctor*);
  static int run();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
