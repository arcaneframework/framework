// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryAdder.h                                         (C) 2000-2024 */
/*                                                                           */
/* Interface de classe permettant d'ajouter un historique de valeur lié à    */
/* un maillage.                                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ITIMEHISTORYMNGADDER_H
#define ARCANE_ITIMEHISTORYMNGADDER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/core/ITimeHistoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryAdder
{
 public:
  virtual ~ITimeHistoryAdder() = default; //!< Libère les ressources

 public:
  virtual void addValue(const TimeHistoryAddValueArg& thp, Real value) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

