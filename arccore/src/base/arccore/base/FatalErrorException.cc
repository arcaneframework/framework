// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FatalErrorException.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Exception lorsqu'une erreur fatale est survenue.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const String& awhere)
: Exception("FatalError",awhere)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const String& awhere,const String& amessage)
: Exception("FatalError",awhere,amessage)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const TraceInfo& awhere)
: Exception("FatalError",awhere)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const TraceInfo& awhere,const String& amessage)
: Exception("FatalError",awhere,amessage)
{
  arccoreDebugPause("FatalError");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FatalErrorException::
FatalErrorException(const FatalErrorException& rhs) ARCCORE_NOEXCEPT
: Exception(rhs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FatalErrorException::
explain(std::ostream& m) const
{
  m << "Fatal error occured.\n"
    << "Can not further proceed.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

