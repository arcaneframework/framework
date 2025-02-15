// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2MpiProfiling.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Implementation de l'interface IMpiProfiling permettant l'instrumentation  */
/* au format OTF2                              .                             */
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiEnum.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/std/internal/Otf2MpiProfiling.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace MessagePassing::Mpi;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Helper sur la fonction de taille de type MPI...
namespace
{
using ReturnType = Otf2MpiProfiling::ReturnType;

uint64_t
getSizeOfMpiType(MPI_Datatype datatype)
{
  int tmp_size(0);
  MPI_Type_size(datatype, &tmp_size);
  return static_cast<uint64_t>(tmp_size);
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Constructeur.
// Note : On ne passe pas le IMpiProfiling a decorer, on appel directement MPI
// pour eviter un un autre appel inutile. Mais on peut facilement le changer le
// cas echeant en rajoutant un IMpiProfiling dans les parametres du ctor.
Otf2MpiProfiling::
Otf2MpiProfiling(Otf2LibWrapper* otf2_wrapper)
: m_otf2_wrapper(otf2_wrapper)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Bcast
ReturnType Otf2MpiProfiling::
broadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
	// calcul des tailles des donnees echangees
	const uint64_t byte_send(m_otf2_wrapper->getMpiRank()==root?getSizeOfMpiType(datatype)*count:0);
	const uint64_t byte_recv(m_otf2_wrapper->getMpiRank()==root?0:getSizeOfMpiType(datatype)*count);

  _doEventEnter(eMpiName::Bcast);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

	int r = MPI_Bcast(buffer, count, datatype, root, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_BCAST, 0 /* comm region */, root,
	                                byte_send /* bytes provided */, byte_recv /* bytes obtained */);

  _doEventLeave(eMpiName::Bcast);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Gather
ReturnType Otf2MpiProfiling::
gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	     int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
	// calcul des tailles des donnees echangees
	const uint64_t byte_send(getSizeOfMpiType(sendtype) * sendcount);
	const uint64_t byte_recv(m_otf2_wrapper->getMpiRank()==root?getSizeOfMpiType(recvtype)*recvcount:0);

  _doEventEnter(eMpiName::Gather);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

	int r = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_GATHER, 0 /* comm region */, root,
	                                byte_send /* bytes provided */, byte_recv /* bytes obtained */);

  _doEventLeave(eMpiName::Gather);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Gatherv
ReturnType Otf2MpiProfiling::
gatherVariable(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
               const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
	// calcul des tailles des donnees echangees
	const uint64_t byte_send(getSizeOfMpiType(sendtype) * sendcount);
	uint64_t byte_recv(0);

	auto f_byte_recv = [&](){for(int i(0); i < m_otf2_wrapper->getMpiNbRank(); ++i)
		                         byte_recv += recvcounts[i] * getSizeOfMpiType(recvtype);};
	if (m_otf2_wrapper->getMpiRank() == root)
		f_byte_recv();

  _doEventEnter(eMpiName::Gatherv);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

  int r = MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_GATHERV, 0 /* comm region */, root,
	                                byte_send /* bytes provided */, byte_recv /* bytes obtained */);

  _doEventLeave(eMpiName::Gatherv);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Allgather
ReturnType Otf2MpiProfiling::
allGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
          int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
	// calcul des tailles des donnees echangees
	const uint64_t byte_send(getSizeOfMpiType(sendtype) * sendcount);
	const uint64_t byte_recv(getSizeOfMpiType(recvtype) * recvcount * m_otf2_wrapper->getMpiNbRank());

  _doEventEnter(eMpiName::Allgather);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

  int r = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_ALLGATHER, 0 /* comm region */, OTF2_UNDEFINED_UINT32 /* root */,
	                                byte_send /* bytes provided */, byte_recv /* bytes obtained */);

  _doEventLeave(eMpiName::Allgather);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Allgatherv
ReturnType Otf2MpiProfiling::
allGatherVariable(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
{
	// calcul des tailles des donnees echangees
	const uint64_t byte_send(getSizeOfMpiType(sendtype) * sendcount);
	uint64_t byte_recv(0);

	for (int i(0); i < m_otf2_wrapper->getMpiNbRank(); ++i)
		byte_recv += recvcounts[i] * getSizeOfMpiType(recvtype);

  _doEventEnter(eMpiName::Allgatherv);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

  int r = MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_ALLGATHERV, 0 /* comm region */, OTF2_UNDEFINED_UINT32 /* root */,
	                                byte_send /* bytes provided */, byte_recv /* bytes obtained */);

  _doEventLeave(eMpiName::Allgatherv);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Scatterv
ReturnType Otf2MpiProfiling::
scatterVariable(const void *sendbuf, const int *sendcounts, const int *displs,
                MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
{
	// calcul des tailles des donnees echangees
	const uint64_t byte_recv(getSizeOfMpiType(recvtype) * recvcount);
	uint64_t byte_send(0);

	auto f_byte_send = [&](){for(int i(0); i < m_otf2_wrapper->getMpiNbRank(); ++i)
		                         byte_send += sendcounts[i];
		                       byte_send *= getSizeOfMpiType(recvtype);};
	if (m_otf2_wrapper->getMpiRank() == root)
		f_byte_send();

  _doEventEnter(eMpiName::Scatterv);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

  int r = MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_SCATTERV, 0 /* comm region */, root,
	                                byte_send /* bytes provided */, byte_recv /* bytes obtained */);

  _doEventLeave(eMpiName::Scatterv);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Alltoall
ReturnType Otf2MpiProfiling::
allToAll(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
         int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  _doEventEnter(eMpiName::Alltoall);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

	int r = MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_ALLTOALL, 0 /* comm region */, OTF2_UNDEFINED_UINT32 /* root */,
	                                0 /* bytes provided */, 0 /* bytes obtained */);

  _doEventLeave(eMpiName::Alltoall);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Alltoallv
ReturnType Otf2MpiProfiling::
allToAllVariable(const void *sendbuf, const int *sendcounts, const int *sdispls,
                 MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
                 const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  _doEventEnter(eMpiName::Alltoallv);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

  int r = MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_ALLTOALLV, 0 /* comm region */, OTF2_UNDEFINED_UINT32 /* root */,
	                                0 /* bytes provided */, 0 /* bytes obtained */);

  _doEventLeave(eMpiName::Alltoallv);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Barrier
ReturnType Otf2MpiProfiling::
barrier(MPI_Comm comm)
{
  _doEventEnter(eMpiName::Barrier);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

  int r = MPI_Barrier(comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_BARRIER, 0 /* comm region */, OTF2_UNDEFINED_UINT32 /* root */,
	                                0 /* bytes provided */, 0 /* bytes obtained */);

  _doEventLeave(eMpiName::Barrier);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Reduce
ReturnType Otf2MpiProfiling::
reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
		   MPI_Op op, int root, MPI_Comm comm)
{
  _doEventEnter(eMpiName::Reduce);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

  int r = MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);

	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_REDUCE, 0 /* comm region */, root,
	                                0 /* bytes provided */, 0 /* bytes obtained */);

  _doEventLeave(eMpiName::Reduce);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Allreduce
ReturnType Otf2MpiProfiling::
allReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	// calcul des tailles des donnees echangees
	const uint64_t byte_send(getSizeOfMpiType(datatype) * count);
	const uint64_t byte_recv(getSizeOfMpiType(datatype) * count);

  _doEventEnter(eMpiName::Allreduce);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

	int r = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

  OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_ALLREDUCE, 0 /* comm region */, OTF2_UNDEFINED_UINT32 /* root */,
	                                byte_send /* bytes provided */, byte_recv /* bytes obtained */);

  _doEventLeave(eMpiName::Allreduce);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Scan
ReturnType Otf2MpiProfiling::
scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  _doEventEnter(eMpiName::Scan);

	OTF2_EvtWriter_MpiCollectiveBegin(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime());

	int r = MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);

	int type_size;
	MPI_Type_size(datatype, &type_size);
	const uint64_t size = static_cast<uint64_t>(type_size) * static_cast<uint64_t>(count);
	OTF2_EvtWriter_MpiCollectiveEnd(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                                OTF2_COLLECTIVE_OP_SCAN, 0 /* comm region */, OTF2_UNDEFINED_UINT32 /* root */,
	                                size /* bytes provided */, size /* bytes obtained */);

  _doEventLeave(eMpiName::Scan);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Sendrecv
ReturnType Otf2MpiProfiling::
sendRecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
         int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
         int source, int recvtag, MPI_Comm comm, MPI_Status *status)
{
  _doEventEnter(eMpiName::Sendrecv);

	int r = MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
			         source, recvtag, comm, status);

  _doEventLeave(eMpiName::Sendrecv);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Isend
ReturnType Otf2MpiProfiling::
iSend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	    MPI_Comm comm, MPI_Request *request)
{
  _doEventEnter(eMpiName::Isend);

	int r = MPI_Isend(buf, count, datatype, dest, tag, comm, request);
  // Comme OTF2 a besoin d'un entier, utilise la valeur fortran de la requête qui
  // est garantie être un entier (contrairement à MPI_Request)
  uint64_t request_id = MPI_Request_c2f(*request);
	OTF2_EvtWriter_MpiIsend(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                        dest, 0 /* comm region */, tag, getSizeOfMpiType(datatype) * count, request_id);

  _doEventLeave(eMpiName::Isend);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Send
ReturnType Otf2MpiProfiling::
send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  _doEventEnter(eMpiName::Send);

  int r = MPI_Send(buf, count, datatype, dest, tag, comm);

	OTF2_EvtWriter_MpiSend(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                       dest, 0 /* comm region */, tag, getSizeOfMpiType(datatype) * count);

  _doEventLeave(eMpiName::Send);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Irecv
ReturnType Otf2MpiProfiling::
iRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
      MPI_Comm comm, MPI_Request *request)
{
  _doEventEnter(eMpiName::Irecv);

	int r = MPI_Irecv(buf, count, datatype, source, tag, comm, request);

	// TODO: arriver a faire fonctionner ce truc...
//	OTF2_EvtWriter_MpiIrecv(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
//	                        source, 0 /* comm region */, tag, getSizeOfMpiType(datatype) * count, *request);
  _doEventLeave(eMpiName::Irecv);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_recv
ReturnType Otf2MpiProfiling::
recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  _doEventEnter(eMpiName::Recv);
	int r = MPI_Recv(buf, count, datatype, source, tag, comm, status);

	OTF2_EvtWriter_MpiRecv(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
	                       source, 0 /* comm region */, tag, getSizeOfMpiType(datatype) * count);

  _doEventLeave(eMpiName::Recv);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Test
ReturnType Otf2MpiProfiling::
test(MPI_Request *request, int *flag, MPI_Status *status)
{
  _doEventEnter(eMpiName::Test);
	if (!MPI_Test(request, flag, status)){
    uint64_t request_id = MPI_Request_c2f(*request);
		OTF2_EvtWriter_MpiRequestTest(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(), request_id);
  }
  _doEventLeave(eMpiName::Test);
  return _ret(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Probe
ReturnType Otf2MpiProfiling::
probe(int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  _doEventEnter(eMpiName::Probe);
	int r = MPI_Probe(source, tag, comm, status);
  _doEventLeave(eMpiName::Probe);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Get_count
ReturnType Otf2MpiProfiling::
getCount(const MPI_Status *status, MPI_Datatype datatype, int *count)
{
  _doEventEnter(eMpiName::Get_count);
  int r = MPI_Get_count(status, datatype, count);
  _doEventLeave(eMpiName::Get_count);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Wait
ReturnType Otf2MpiProfiling::
wait(MPI_Request *request, MPI_Status *status)
{
  _doEventEnter(eMpiName::Wait);
  int r = MPI_Wait(request, status);
  _doEventLeave(eMpiName::Wait);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Waitall
ReturnType Otf2MpiProfiling::
waitAll(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
{
  _doEventEnter(eMpiName::Waitall);
	int r = MPI_Waitall(count, array_of_requests, array_of_statuses);
  _doEventLeave(eMpiName::Waitall);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Testsome
ReturnType Otf2MpiProfiling::
testSome(int incount, MPI_Request array_of_requests[], int *outcount,
         int array_of_indices[], MPI_Status array_of_statuses[])
{
  _doEventEnter(eMpiName::Testsome);
	int r = MPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
  _doEventLeave(eMpiName::Testsome);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! MPI_Waitsome
ReturnType Otf2MpiProfiling::
waitSome(int incount, MPI_Request array_of_requests[], int *outcount,
         int array_of_indices[], MPI_Status array_of_statuses[])
{
  _doEventEnter(eMpiName::Waitsome);
  int r = MPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
  _doEventLeave(eMpiName::Waitsome);
  return _ret(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MpiProfiling::
_doEventEnter(eMpiName event_name)
{
  OTF2_EvtWriter_Enter(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
                       static_cast<uint32_t>(event_name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MpiProfiling::
_doEventLeave(eMpiName event_name)
{
  OTF2_EvtWriter_Leave(m_otf2_wrapper->getEventWriter(), NULL, Otf2LibWrapper::getTime(),
                       static_cast<uint32_t>(event_name));

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}  // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
