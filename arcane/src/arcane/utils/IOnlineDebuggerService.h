// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHyodaService.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface of a hybrid debugger service.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IHYODA_SERVICE_H
#define ARCANE_IHYODA_SERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Move this file to 'arcane/core' because it uses types
// from 'arcane/core'

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
 * \brief Interface of a hybrid debugger service
 *
 * \note In development
 */
class IOnlineDebuggerService
{
 public:

  virtual ~IOnlineDebuggerService() {}

 public:

  virtual Real loopbreak(ISubDomain*) = 0;
  virtual Real softbreak(ISubDomain*, const char*, const char*, int) = 0;
  virtual void hook(ISubDomain*, Real) = 0;
  virtual void ijval(int, int, int*, int*, double*) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
