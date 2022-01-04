// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Assertion.cc                                                (C) 2000-2020 */
/*                                                                           */
/* Ensemble d'assertions utilisées pour les tests unitaires.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Assertion.h"

#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

void Assertion::
_checkAssertion(bool is_error,const TraceInfo& where,
                const String& expected, const String& actual, IParallelMng* pm)
{
  bool global_error = is_error;
  if (pm){
    Int32 e = (is_error) ? 1 : 0;
    Int32 ge = pm->reduce(Parallel::ReduceSum,e);
    global_error = (ge!=0);
  }
  if (global_error)
    throw AssertionException(where, expected, actual);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
