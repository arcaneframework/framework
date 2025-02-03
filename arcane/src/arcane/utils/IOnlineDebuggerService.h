// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHyodaService.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service de debugger hybrid.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IHYODA_SERVICE_H
#define ARCANE_IHYODA_SERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Mettre ce fichier dans 'arcane/core' car il utilise des types
// de 'arcane/core'

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service de debugger hybrid
 *
 * \note En cours de développement
 */
class IOnlineDebuggerService{
 public:
  virtual ~IOnlineDebuggerService(){}
 public:
  virtual Real loopbreak(ISubDomain*)=0;
  virtual Real softbreak(ISubDomain*,const char*,const char*,int)=0;
  virtual void hook(ISubDomain*,Real)=0;
  virtual void ijval(int,int,int*,int*,double*)=0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
