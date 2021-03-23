// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHyodaService.h                                             (C) 2000-2011 */
/*                                                                           */
/* Interface d'un service de debugger hybrid.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IHYODA_SERVICE_H
#define ARCANE_IHYODA_SERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/ISubDomain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
