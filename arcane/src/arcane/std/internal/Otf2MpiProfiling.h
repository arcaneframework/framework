// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2MpiProfiling.h                                          (C) 2000-2025 */
/*                                                                           */
/* Implementation de l'interface IMpiProfiling permettant l'instrumentation  */
/* au format OTF2                              .                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_OTF2MPIPROFILING_H
#define ARCANE_STD_INTERNAL_OTF2MPIPROFILING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/collections/CollectionsGlobal.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing_mpi/internal/IMpiProfiling.h"
#include "arccore/message_passing_mpi/MessagePassingMpiEnum.h"
#include "arcane/std/internal/Otf2LibWrapper.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace MessagePassing::Mpi;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implementation de l'interface des operations MPI.
 * Decore chacun des appels MPI avec les fonctions de la librairie
 * Otf2 pour faire du profiling.
 */
class Otf2MpiProfiling
: public IMpiProfiling
{
 public:
  // Pour l'instant void pour des raisons de compatibilité mais devra à terme
  // être IMpiProfiling::ReturnType
  using ReturnType = void;

 public:
  explicit Otf2MpiProfiling(Otf2LibWrapper* otf2_wrapper);
  ~Otf2MpiProfiling() override = default;

 public:
  // Bcast
  ReturnType broadcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) final;
  // Gather
  ReturnType gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                    int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) final;
  // Gatherv
  ReturnType gatherVariable(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                            const int* recvcounts, const int* displs, MPI_Datatype recvtype, int root, MPI_Comm comm) final;
  // Allgather
  ReturnType allGather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                       int recvcount, MPI_Datatype recvtype, MPI_Comm comm) final;
  // Allgatherv
  ReturnType allGatherVariable(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                               const int* recvcounts, const int* displs, MPI_Datatype recvtype, MPI_Comm comm) final;
  // Scatterv
  ReturnType scatterVariable(const void* sendbuf, const int* sendcounts, const int* displs,
                             MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm) final;
  // Alltoall
  ReturnType allToAll(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                      int recvcount, MPI_Datatype recvtype, MPI_Comm comm) final;
  // Alltoallv
  ReturnType allToAllVariable(const void* sendbuf, const int* sendcounts, const int* sdispls,
                              MPI_Datatype sendtype, void* recvbuf, const int* recvcounts,
                              const int* rdispls, MPI_Datatype recvtype, MPI_Comm comm) final;
  // Barrier
  ReturnType barrier(MPI_Comm comm) final;
  // Reduce
  ReturnType reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, int root, MPI_Comm comm) final;
  // Allreduce
  ReturnType allReduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm) final;
  // Scan
  ReturnType scan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) final;
  // Sendrecv
  ReturnType sendRecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                      int sendtag, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag, MPI_Comm comm, MPI_Status* status) final;
  // Isend
  ReturnType iSend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request* request) final;
  // Send
  ReturnType send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) final;
  // Irecv
  ReturnType iRecv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
                   MPI_Comm comm, MPI_Request* request) final;
  // recv
  ReturnType recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status) final;
  // Test
  ReturnType test(MPI_Request* request, int* flag, MPI_Status* status) final;
  // Probe
  ReturnType probe(int source, int tag, MPI_Comm comm, MPI_Status* status) final;
  // Get_count
  ReturnType getCount(const MPI_Status* status, MPI_Datatype datatype, int* count) final;
  // Wait
  ReturnType wait(MPI_Request* request, MPI_Status* status) final;
  // Waitall
  ReturnType waitAll(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) final;
  // Testsome
  ReturnType testSome(int incount, MPI_Request array_of_requests[], int* outcount,
                      int array_of_indices[], MPI_Status array_of_statuses[]) final;
  // Waitsome
  ReturnType waitSome(int incount, MPI_Request array_of_requests[], int* outcount,
                      int array_of_indices[], MPI_Status array_of_statuses[]) final;

 private:
  Otf2LibWrapper* m_otf2_wrapper;

 private:
  void _doEventEnter(eMpiName event_name);
  void _doEventLeave(eMpiName event_name);
  ReturnType _ret(int r) const
  {
    return (ReturnType)(r);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
