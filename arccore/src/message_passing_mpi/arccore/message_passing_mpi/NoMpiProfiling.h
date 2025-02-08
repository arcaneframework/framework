// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NoMpiProfiling.h                                            (C) 2000-2025 */
/*                                                                           */
/* Implementation de l'interface IMpiProfiling.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_NOMPIPROFILING_H
#define ARCCORE_MESSAGEPASSINGMPI_NOMPIPROFILING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing_mpi/IMpiProfiling.h"
#include "arccore/message_passing/Request.h"
#include "arccore/collections/CollectionsGlobal.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implementation de l'interface des operations MPI.
 * Correspond a un simple appel aux fonctions MPI du meme nom
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT NoMpiProfiling
: public IMpiProfiling
{
 public:
  NoMpiProfiling() = default;
  virtual ~NoMpiProfiling() = default;

  ReturnType _ret(int r)
  {
    return (ReturnType)(r);
  }

 public:
  // Bcast
  ReturnType broadcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) final
  {
    return _ret(MPI_Bcast(buffer, count, datatype, root, comm));
  }
  // Gather
  ReturnType gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                    int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) final
  {
    return _ret(MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm));
  }
  // Gatherv
  ReturnType gatherVariable(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                            const int* recvcounts, const int* displs, MPI_Datatype recvtype, int root, MPI_Comm comm) final
  {
    return _ret(MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm));
  }
  // allGather
  ReturnType allGather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                       int recvcount, MPI_Datatype recvtype, MPI_Comm comm) final
  {
    return _ret(MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
  }
  // Allgatherv
  ReturnType allGatherVariable(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                               const int* recvcounts, const int* displs, MPI_Datatype recvtype, MPI_Comm comm) final
  {
    return _ret(MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm));
  }
  // Scatterv
  ReturnType scatterVariable(const void* sendbuf, const int* sendcounts, const int* displs,
                             MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm) final
  {
    return _ret(MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm));
  }
  // Alltoall
  ReturnType allToAll(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                      int recvcount, MPI_Datatype recvtype, MPI_Comm comm) final
  {
    return _ret(MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
  }
  // Alltoallv
  ReturnType allToAllVariable(const void* sendbuf, const int* sendcounts, const int* sdispls,
                              MPI_Datatype sendtype, void* recvbuf, const int* recvcounts,
                              const int* rdispls, MPI_Datatype recvtype, MPI_Comm comm) final
  {
    return _ret(MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm));
  }
  // Barrier
  ReturnType barrier(MPI_Comm comm) final
  {
    return _ret(MPI_Barrier(comm));
  }
  // Reduce
  ReturnType reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, int root, MPI_Comm comm) final
  {
    return _ret(MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm));
  }
  // Allreduce
  ReturnType allReduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm) final
  {
    return _ret(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm));
  }
  // Scan
  ReturnType scan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) final
  {
    return _ret(MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm));
  }
  // Sendrecv
  ReturnType sendRecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                      int sendtag, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag, MPI_Comm comm, MPI_Status* status) final
  {
    return _ret(MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
                             source, recvtag, comm, status));
  }
  // Isend
  ReturnType iSend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request* request) final
  {
    return _ret(MPI_Isend(buf, count, datatype, dest, tag, comm, request));
  }
  // Send
  ReturnType send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) final
  {
    return _ret(MPI_Send(buf, count, datatype, dest, tag, comm));
  }
  // Irecv
  ReturnType iRecv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
                   MPI_Comm comm, MPI_Request* request) final
  {
    return _ret(MPI_Irecv(buf, count, datatype, source, tag, comm, request));
  }
  // recv
  ReturnType recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status) final
  {
    return _ret(MPI_Recv(buf, count, datatype, source, tag, comm, status));
  }
  // Test
  ReturnType test(MPI_Request* request, int* flag, MPI_Status* status) final
  {
    return _ret(MPI_Test(request, flag, status));
  }
  // Probe
  ReturnType probe(int source, int tag, MPI_Comm comm, MPI_Status* status) final
  {
    return _ret(MPI_Probe(source, tag, comm, status));
  }
  // Get_count
  ReturnType getCount(const MPI_Status* status, MPI_Datatype datatype, int* count) final
  {
    return _ret(MPI_Get_count(status, datatype, count));
  }
  // Wait
  ReturnType wait(MPI_Request* request, MPI_Status* status) final
  {
    return _ret(MPI_Wait(request, status));
  }
  // Waitall
  ReturnType waitAll(int count, MPI_Request* array_of_requests, MPI_Status* array_of_statuses) final
  {
    return _ret(MPI_Waitall(count, array_of_requests, array_of_statuses));
  }
  // Testsome
  ReturnType testSome(int incount, MPI_Request* array_of_requests, int* outcount,
                      int* array_of_indices, MPI_Status* array_of_statuses) final
  {
    return _ret(MPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses));
  }
  // Waitsome
  ReturnType waitSome(int incount, MPI_Request* array_of_requests, int* outcount,
                      int* array_of_indices, MPI_Status* array_of_statuses) final
  {
    return _ret(MPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
