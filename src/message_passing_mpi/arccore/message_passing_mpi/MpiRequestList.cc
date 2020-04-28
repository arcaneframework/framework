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
/* MpiRequestList.cc                                           (C) 2000-2020 */
/*                                                                           */
/* Classe de base d'une liste de requêtes.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiRequestList.h"
#include "arccore/message_passing_mpi/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiRequestList::
_wait(eWaitType wait_type)
{
  switch(wait_type){
  case WaitAll:
    m_adapter->waitAllRequests(_requests());
    break;
  case WaitSome:
    _doWaitSome(false);
    break;
  case WaitSomeNonBlocking:
    _doWaitSome(true);
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiRequestList::
_doWaitSome(bool is_non_blocking)
{
  Integer nb_request = size();
  m_requests_status.resize(nb_request);
  m_adapter->waitSomeRequests(_requests(),_requestsDone(), is_non_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
