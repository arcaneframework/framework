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

Integer MpiRequestList::
wait(eWaitType wait_type)
{
  removeDoneRequests();
  switch(wait_type){
  case WaitAll:
    m_adapter->waitAllRequests(_requests());
    return (-1);
  case WaitSome:
    return _doWaitSome(false);
  case WaitSomeNonBlocking:
    return _doWaitSome(true);
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiRequestList::
_doWaitSome(bool is_non_blocking)
{
  Integer nb_request = nbRequest();
  m_requests_status.resize(nb_request);
  m_adapter->waitSomeRequests(_requests(),_requestsDone(), is_non_blocking);
  Integer nb_done = 0;
  for( bool v : _requestsDone() )
    if (v)
      ++nb_done;
  return nb_done;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
