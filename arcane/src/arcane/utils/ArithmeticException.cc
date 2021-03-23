// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArithmeticException.cc                                      (C) 2000-2016 */
/*                                                                           */
/* Exception lorsqu'une erreur arithmétique survient.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArithmeticException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& awhere)
: Exception("ArithmeticException",awhere,"arithmetic or floating error")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& awhere,const StackTrace& stack_trace)
: Exception("ArithmeticException",awhere,"arithmetic or floating error",stack_trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& awhere,const String& message)
: Exception("ArithmeticException",awhere,message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& where,const String& message,
                    const StackTrace& stack_trace)
  : Exception("ArithmeticException",where,message,stack_trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

