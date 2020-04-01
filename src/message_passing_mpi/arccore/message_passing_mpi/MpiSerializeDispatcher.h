// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiSerializeDispatcher.h                                    (C) 2000-2020 */
/*                                                                           */
/* Gestion des messages de sérialisation avec MPI.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEDISPATCHER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/ISerializeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
class MpiSerializeDispatcher
: public ISerializeDispatcher
{
 public:
  MpiSerializeDispatcher(IMessagePassingMng* parallel_mng, MpiAdapter* adapter);

 public:

 private:
  IMessagePassingMng* m_parallel_mng;
  MpiAdapter* m_adapter;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
