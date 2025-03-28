// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RequestListBase.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Liste de requêtes MPI.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/RequestListBase.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RequestListBase::
clear()
{
  m_requests.clear();
  m_requests_done.clear();
  m_done_request_indexes.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RequestListBase::
removeDoneRequests()
{
  // TODO: optimiser cela en supprimant plusieurs valeurs d'un coup
  for( Integer i=0, n=m_requests.size(); i<n; ++i ){
    if (m_requests_done[i]){
      _removeRequestAtIndex(i);
      --i;
      --n;      
    }
  }
  m_done_request_indexes.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> RequestListBase::
doneRequestIndexes() const
{
  return m_done_request_indexes.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 RequestListBase::
wait(eWaitType wait_type)
{
  m_requests_done.fill(false);

  // Délegue l'appel effectif à la classe dérivée qui doit remplir
  // le champ \a m_requests_done (sauf pour WaitAll)
  _wait(wait_type);

  if (wait_type==WaitAll)
    m_requests_done.fill(true);

  Integer nb_request = size();
  m_done_request_indexes.clear();
  m_done_request_indexes.reserve(nb_request);
  Int32 nb_done = 0;
  for( Integer i=0; i<nb_request; ++i )
    if (m_requests_done[i]){
      m_done_request_indexes.add(i);
      ++nb_done;
    }
  return nb_done;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
