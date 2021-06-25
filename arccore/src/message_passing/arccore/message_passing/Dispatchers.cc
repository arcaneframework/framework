// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
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
