/*---------------------------------------------------------------------------*/
/* Dispatchers.cc                                              (C) 2000-2018 */
/*                                                                           */
/* Conteneur des dispatchers.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Dispatchers::
Dispatchers()
: m_char(nullptr)
, m_unsigned_char(nullptr)
, m_signed_char(nullptr)
, m_short(nullptr)
, m_unsigned_short(nullptr)
, m_int(nullptr)
, m_unsigned_int(nullptr)
, m_long(nullptr)
, m_unsigned_long(nullptr)
, m_long_long(nullptr)
, m_unsigned_long_long(nullptr)
, m_float(nullptr)
, m_double(nullptr)
, m_long_double(nullptr)
, m_is_delete_dispatchers(false)
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
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
