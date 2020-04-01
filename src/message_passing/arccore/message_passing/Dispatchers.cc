// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Dispatchers.cc                                              (C) 2000-2020 */
/*                                                                           */
/* Conteneur des dispatchers.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/IControlDispatcher.h"

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
    delete m_control;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
