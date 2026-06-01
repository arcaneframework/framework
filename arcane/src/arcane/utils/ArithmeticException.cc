// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArithmeticException.cc                                      (C) 2000-2016 */
/*                                                                           */
/* Exception when an arithmetic error occurs.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArithmeticException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& awhere)
: Exception("ArithmeticException", awhere, "arithmetic or floating error")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& awhere, const StackTrace& stack_trace)
: Exception("ArithmeticException", awhere, "arithmetic or floating error", stack_trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& awhere, const String& message)
: Exception("ArithmeticException", awhere, message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArithmeticException::
ArithmeticException(const TraceInfo& where, const String& message,
                    const StackTrace& stack_trace)
: Exception("ArithmeticException", where, message, stack_trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
