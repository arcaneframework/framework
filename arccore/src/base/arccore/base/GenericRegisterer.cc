// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericRegisterer.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Enregistreur générique de types globaux.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/GenericRegisterer.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericRegistererBase::
doErrorConflict()
{
  // Cette fonction peut-être appelée lors des constructeurs globaux.
  // On évite donc qu'elle lance des exceptions et on fait directement un abort.
  std::cerr << "Arcane Fatal Error: Service conflict in service registration" << std::endl;
  abort();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericRegistererBase::
doErrorNonZeroCount()
{
  // Cette fonction peut-être appelée lors des constructeurs globaux.
  // On évite donc qu'elle lance des exceptions et on fait directement un abort.
  std::cerr << "Arcane Fatal Error: Service breaks service registration (inconsistent shortcut)" << std::endl;
  abort();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

