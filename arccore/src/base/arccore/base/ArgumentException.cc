// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArgumentException.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Exception lorsqu'un argument est invalide.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/ArgumentException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const String& awhere)
: Exception("ArgumentException",awhere,"Bad argument")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const String& awhere,const String& amessage)
: Exception("ArgumentException",awhere,amessage)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const TraceInfo& awhere)
: Exception("ArgumentException",awhere,"Bad argument")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const TraceInfo& awhere,const String& amessage)
: Exception("ArgumentException",awhere,amessage)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const ArgumentException& rhs) ARCCORE_NOEXCEPT
: Exception(rhs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
~ArgumentException() ARCCORE_NOEXCEPT
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

