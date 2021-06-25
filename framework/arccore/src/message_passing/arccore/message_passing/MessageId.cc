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
/* MessageId.cc                                                (C) 2000-2020 */
/*                                                                           */
/* Identifiant d'un message point à point.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessageId.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessagePassing::MessageId::_Message MessagePassing::MessageId::null_message;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MessagePassing::MessageId::
print(std::ostream& o) const
{
  o << "(id=";
  if (m_type==T_Null)
    o << "(null)";
  else if (m_type==T_Int)
    o << m_message.i;
  else if (m_type==T_Long)
    o << m_message.l;
  else
    o << m_message.cv;
  o << ",source_rank=" << m_source_info.rank()
    << " tag=" << m_source_info.tag()
    << " size=" << m_source_info.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
