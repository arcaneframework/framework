// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadCastException.cc                                         (C) 2000-2016 */
/*                                                                           */
/* Exception lorsqu'une conversion est invalide.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/BadCastException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadCastException::
BadCastException(const String& awhere)
: Exception("BadCastException",awhere,"Bad argument")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadCastException::
BadCastException(const String& awhere,const String& amessage)
: Exception("BadCastException",awhere,amessage)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadCastException::
BadCastException(const TraceInfo& awhere)
: Exception("BadCastException",awhere,"Bad argument")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadCastException::
BadCastException(const TraceInfo& awhere,const String& amessage)
: Exception("BadCastException",awhere,amessage)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

