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
/* IMpiProfiling.h                                             (C) 2000-2018 */
/*                                                                           */
/* Interface d'abstraction des operations MPI.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_IMPIPROFILING_H
#define ARCCORE_MESSAGEPASSINGMPI_IMPIPROFILING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include "arccore/collections/CollectionsGlobal.h"

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{
namespace Mpi
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
{
 public:

	IMpiProfiling() = default;
  virtual ~IMpiProfiling() = default;

 public:
	// Bcast
	virtual void broadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) = 0;
	// Gather
	virtual void gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	                    int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) = 0;
	// Gatherv
	virtual void gatherVariable(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	                            const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root,
	                            MPI_Comm comm) = 0;
	// allGather
	virtual void allGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	                       int recvcount, MPI_Datatype recvtype, MPI_Comm comm) = 0;
	// Allgatherv
	virtual void allGatherVariable(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	                               const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm) = 0;
	// Scatterv
	virtual void scatterVariable(const void *sendbuf, const int *sendcounts, const int *displs,
	                             MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	                             int root, MPI_Comm comm) = 0;
	// Alltoall
	virtual void allToAll(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	                      int recvcount, MPI_Datatype recvtype, MPI_Comm comm) = 0;
	// Alltoallv
	virtual void allToAllVariable(const void *sendbuf, const int *sendcounts, const int *sdispls,
	                              MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
	                              const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) = 0;
	// Barrier
	virtual void barrier(MPI_Comm comm) = 0;
	// Reduce
	virtual void reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	                    MPI_Op op, int root, MPI_Comm comm) = 0;
	// Allreduce
	virtual void allReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	                       MPI_Op op, MPI_Comm comm) = 0;
	// Scan
	virtual void scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
	                  MPI_Comm comm) = 0;
	// Sendrecv
	virtual void sendRecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
	                      int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	                      int source, int recvtag, MPI_Comm comm, MPI_Status *status) = 0;
	// Isend
	virtual void iSend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	                   MPI_Comm comm, MPI_Request *request) = 0;
	// Send
	virtual void send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) = 0;
  // Irecv
  virtual void iRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                     MPI_Comm comm, MPI_Request *request) = 0;
	// recv
	virtual void recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
	                  MPI_Status *status) = 0;
	// Test
	virtual void test(MPI_Request *request, int *flag, MPI_Status *status) = 0;
	// Probe
	virtual void probe(int source, int tag, MPI_Comm comm, MPI_Status *status) = 0;
	// Get_count
	virtual void getCount(const MPI_Status *status, MPI_Datatype datatype, int *count) = 0;
	// Wait
	virtual void wait(MPI_Request *request, MPI_Status *status) = 0;
	// Waitall
	virtual void waitAll(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses) = 0;
	// Testsome
	virtual void testSome(int incount, MPI_Request *array_of_requests, int *outcount,
	                      int *array_of_indices, MPI_Status *array_of_statuses) = 0;
	// Waitsome
	virtual void waitSome(int incount, MPI_Request *array_of_requests, int *outcount,
	                      int *array_of_indices, MPI_Status *array_of_statuses) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
