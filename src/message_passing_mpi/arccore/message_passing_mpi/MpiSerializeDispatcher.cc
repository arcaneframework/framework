// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiSerializeDispatcher.cc                                   (C) 2000-2020 */
/*                                                                           */
/* Gestion des messages de sérialisation avec MPI.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiSerializeDispatcher.h"
#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"
#include "arccore/message_passing/Request.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiSerializeDispatcher::
MpiSerializeDispatcher(IMessagePassingMng* parallel_mng, MpiAdapter* adapter)
: m_parallel_mng(parallel_mng)
, m_adapter(adapter)
{
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
