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
/* NoMpiProfiling.h                                            (C) 2000-2018 */
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

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implementation de l'interface des operations MPI.
 * Correspond a un simple appel aux fonctions MPI du meme nom
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT NoMpiProfiling : public IMpiProfiling
{
 public:
  NoMpiProfiling() = default;
  virtual ~NoMpiProfiling() = default;

 public:
	// Bcast
	void broadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) final {
		MPI_Bcast(buffer, count, datatype, root, comm);
	}
	// Gather
	void gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	            int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) final {
		MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
	}
	// Gatherv
	void gatherVariable(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	                    const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm) final {
		MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
	}
	// allGather
	void allGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	               int recvcount, MPI_Datatype recvtype, MPI_Comm comm) final {
		MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
	}
	// Allgatherv
	void allGatherVariable(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	                       const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm) final {
		MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
	}
	// Scatterv
	void scatterVariable(const void *sendbuf, const int *sendcounts, const int *displs,
	                     MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	                     int root, MPI_Comm comm) final {
		MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
	}
	// Alltoall
	void allToAll(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
	              int recvcount, MPI_Datatype recvtype, MPI_Comm comm) final {
		MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
	}
	// Alltoallv
	void allToAllVariable(const void *sendbuf, const int *sendcounts, const int *sdispls,
	                      MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
	                      const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) final {
		MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
	}
	// Barrier
	void barrier(MPI_Comm comm) final {
		MPI_Barrier(comm);
	}
	// Reduce
	void reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	            MPI_Op op, int root, MPI_Comm comm) final {
		MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
	}
	// Allreduce
	void allReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	               MPI_Op op, MPI_Comm comm) final {
		MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
	}
	// Scan
	void scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) final {
		MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
	}
	// Sendrecv
	void sendRecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
	              int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	              int source, int recvtag, MPI_Comm comm, MPI_Status *status) final {
		MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
				         source, recvtag, comm, status);
	}
	// Isend
	void iSend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	           MPI_Comm comm, MPI_Request *request) final {
		MPI_Isend(buf, count, datatype, dest, tag, comm, request);
	}
	// Send
	void send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) final {
		MPI_Send(buf, count, datatype, dest, tag, comm);
	}
  // Irecv
  void iRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Request *request) final {
		MPI_Irecv(buf, count, datatype, source, tag, comm, request);
  }
	// recv
	void recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) final {
		MPI_Recv(buf, count, datatype, source, tag, comm, status);
	}
	// Test
	void test(MPI_Request *request, int *flag, MPI_Status *status) final {
		MPI_Test(request, flag, status);
	}
	// Probe
	void probe(int source, int tag, MPI_Comm comm, MPI_Status *status) final {
		MPI_Probe(source, tag, comm, status);
	}
	// Get_count
	void getCount(const MPI_Status *status, MPI_Datatype datatype, int *count) final {
		MPI_Get_count(status, datatype, count);
	}
	// Wait
	void wait(MPI_Request *request, MPI_Status *status) final {
		MPI_Wait(request, status);
	}
	// Waitall
	void waitAll(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses) final {
		MPI_Waitall(count, array_of_requests, array_of_statuses);
	}
	// Testsome
  void testSome(int incount, MPI_Request *array_of_requests, int *outcount,
                int *array_of_indices, MPI_Status *array_of_statuses) final {
		MPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
	}
	// Waitsome
	void waitSome(int incount, MPI_Request *array_of_requests, int *outcount,
	              int *array_of_indices, MPI_Status *array_of_statuses) final {
		MPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
	}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
