// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelFatalErrorException.cc                              (C) 2000-2018 */
/*                                                                           */
/* Exception when a 'parallel' fatal error occurred.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/ParallelFatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelFatalErrorException::
ParallelFatalErrorException(const String& where)
: Exception("ParallelFatalError", where)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelFatalErrorException::
ParallelFatalErrorException(const TraceInfo& where)
: Exception("ParallelFatalError", where)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelFatalErrorException::
ParallelFatalErrorException(const String& where, const String& message)
: Exception("ParallelFatalError", where, message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelFatalErrorException::
ParallelFatalErrorException(const TraceInfo& where, const String& message)
: Exception("ParallelFatalError", where, message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelFatalErrorException::
explain(std::ostream& m) const
{
  m << "Fatal error occured.\n"
    << "Can not further proceed.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
