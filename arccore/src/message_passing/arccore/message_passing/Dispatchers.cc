// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Dispatchers.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Conteneur des dispatchers.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/message_passing/Request.h"

#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Dispatchers::
Dispatchers()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Dispatchers::
~Dispatchers()
{
  if (m_is_delete_dispatchers) {
    delete m_char;
    delete m_unsigned_char;
    delete m_signed_char;
    delete m_short;
    delete m_unsigned_short;
    delete m_int;
    delete m_unsigned_int;
    delete m_long;
    delete m_unsigned_long;
    delete m_long_long;
    delete m_unsigned_long_long;
    delete m_float;
    delete m_double;
    delete m_long_double;
    delete m_bfloat16;
    delete m_float16;
    delete m_control;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void
_internalThrowNotImplementedTypeDispatcher ARCCORE_NORETURN ()
{
  ARCCORE_THROW(NotImplementedException,"Generic gather");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
