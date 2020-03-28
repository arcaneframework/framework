// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
