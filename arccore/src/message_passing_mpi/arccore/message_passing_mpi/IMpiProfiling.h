// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMpiProfiling.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface d'abstraction des operations MPI.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_IMPIPROFILING_H
#define ARCCORE_MESSAGEPASSINGMPI_IMPIPROFILING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include "arccore/message_passing/IProfiler.h"

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
 * \brief Interface d'abstraction pour les operations MPI.
 * Sert principalement a utiliser un decorateur pour les fonctions MPI
 * afin de les profiler sans etre trop verbeux et intrusif dans le MPIAdapter
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT IMpiProfiling
: public IProfiler
{
 public:
  // Type de retour. Pour l'instant 'void' pour être conforme avec l'existant
  // mais cela devrait être 'int' car c'est le type de retour de toutes les méthodes MPI.
  using ReturnType = void;

 public:
  IMpiProfiling() = default;
  virtual ~IMpiProfiling() = default;

 public:
  // Bcast
  virtual ReturnType broadcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) = 0;
  // Gather
  virtual ReturnType gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                            int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) = 0;
  // Gatherv
  virtual ReturnType gatherVariable(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                    const int* recvcounts, const int* displs, MPI_Datatype recvtype, int root,
                                    MPI_Comm comm) = 0;
  // allGather
  virtual ReturnType allGather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                               int recvcount, MPI_Datatype recvtype, MPI_Comm comm) = 0;
  // Allgatherv
  virtual ReturnType allGatherVariable(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                       const int* recvcounts, const int* displs, MPI_Datatype recvtype, MPI_Comm comm) = 0;
  // Scatterv
  virtual ReturnType scatterVariable(const void* sendbuf, const int* sendcounts, const int* displs,
                                     MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                                     int root, MPI_Comm comm) = 0;
  // Alltoall
  virtual ReturnType allToAll(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                              int recvcount, MPI_Datatype recvtype, MPI_Comm comm) = 0;
  // Alltoallv
  virtual ReturnType allToAllVariable(const void* sendbuf, const int* sendcounts, const int* sdispls,
                                      MPI_Datatype sendtype, void* recvbuf, const int* recvcounts,
                                      const int* rdispls, MPI_Datatype recvtype, MPI_Comm comm) = 0;
  // Barrier
  virtual ReturnType barrier(MPI_Comm comm) = 0;
  // Reduce
  virtual ReturnType reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                            MPI_Op op, int root, MPI_Comm comm) = 0;
  // Allreduce
  virtual ReturnType allReduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                               MPI_Op op, MPI_Comm comm) = 0;
  // Scan
  virtual ReturnType scan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                          MPI_Comm comm) = 0;
  // Sendrecv
  virtual ReturnType sendRecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                              int sendtag, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                              int source, int recvtag, MPI_Comm comm, MPI_Status* status) = 0;
  // Isend
  virtual ReturnType iSend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
                           MPI_Comm comm, MPI_Request* request) = 0;
  // Send
  virtual ReturnType send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) = 0;
  // Irecv
  virtual ReturnType iRecv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
                           MPI_Comm comm, MPI_Request* request) = 0;
  // recv
  virtual ReturnType recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                          MPI_Status* status) = 0;
  // Test
  virtual ReturnType test(MPI_Request* request, int* flag, MPI_Status* status) = 0;
  // Probe
  virtual ReturnType probe(int source, int tag, MPI_Comm comm, MPI_Status* status) = 0;
  // Get_count
  virtual ReturnType getCount(const MPI_Status* status, MPI_Datatype datatype, int* count) = 0;
  // Wait
  virtual ReturnType wait(MPI_Request* request, MPI_Status* status) = 0;
  // Waitall
  virtual ReturnType waitAll(int count, MPI_Request* array_of_requests, MPI_Status* array_of_statuses) = 0;
  // Testsome
  virtual ReturnType testSome(int incount, MPI_Request* array_of_requests, int* outcount,
                              int* array_of_indices, MPI_Status* array_of_statuses) = 0;
  // Waitsome
  virtual ReturnType waitSome(int incount, MPI_Request* array_of_requests, int* outcount,
                              int* array_of_indices, MPI_Status* array_of_statuses) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
