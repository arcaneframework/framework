// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Dispatchers.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Conteneur des dispatchers.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/Request.h"

#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
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
    m_container.apply([&](auto x){ delete x; });

    delete m_control;
    delete m_serialize;
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
