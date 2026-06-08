// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiAdapter.cc                                               (C) 2000-2026 */
/*                                                                           */
/* Parallelism manager using MPI.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

#include "arccore/trace/ITraceMng.h"

#include "arccore/collections/Array.h"

#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/IStat.h"
#include "arccore/message_passing/internal/SubRequestCompletionInfo.h"

#include "arccore/base/IStackTraceService.h"
#include "arccore/base/TimeoutException.h"
#include "arccore/base/String.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/TraceInfo.h"

#include "arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h"
#include "arccore/message_passing_mpi/internal/MpiLock.h"
#include "arccore/message_passing_mpi/internal/NoMpiProfiling.h"
#include "arccore/message_passing_mpi/internal/MpiRequest.h"
#include "arccore/message_passing_mpi/internal/MpiMachineShMemWinBaseInternalCreator.h"

#include <cstdint>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiAdapter::RequestSet
: public TraceAccessor
{
 public:

  struct RequestInfo
  {
    TraceInfo m_trace;
    String m_stack_trace;
  };

 public:

  typedef std::map<MPI_Request, RequestInfo>::iterator Iterator;

 public:

  explicit RequestSet(ITraceMng* tm)
  : TraceAccessor(tm)
  {
    m_trace_mng_ref = makeRef(tm);
    if (arccoreIsCheck()) {
      m_no_check_request = false;
      m_request_error_is_fatal = true;
    }
    if (Platform::getEnvironmentVariable("ARCCORE_NOREPORT_ERROR_MPIREQUEST") == "TRUE")
      m_is_report_error_in_request = false;
    if (Platform::getEnvironmentVariable("ARCCORE_MPIREQUEST_STACKTRACE") == "TRUE")
      m_use_trace_full_stack = true;
    if (Platform::getEnvironmentVariable("ARCCORE_TRACE_MPIREQUEST") == "TRUE")
      m_trace_mpirequest = true;
  }

 public:

  void addRequest(MPI_Request request)
  {
    if (m_no_check_request)
      return;
    if (m_trace_mpirequest)
      info() << "MpiAdapter: AddRequest r=" << request;
    _addRequest(request, TraceInfo());
  }
  void addRequest(MPI_Request request, const TraceInfo& ti)
  {
    if (m_no_check_request)
      return;
    if (m_trace_mpirequest)
      info() << "MpiAdapter: AddRequest r=" << request;
    _addRequest(request, ti);
  }
  void removeRequest(MPI_Request request)
  {
    if (m_no_check_request)
      return;
    if (m_trace_mpirequest)
      info() << "MpiAdapter: RemoveRequest r=" << request;
    _removeRequest(request);
  }
  void removeRequest(Iterator request_iter)
  {
    if (m_no_check_request)
      return;
    if (request_iter == m_allocated_requests.end()) {
      if (m_trace_mpirequest)
        info() << "MpiAdapter: RemoveRequestIter null iterator";
      return;
    }
    if (m_trace_mpirequest)
      info() << "MpiAdapter: RemoveRequestIter r=" << request_iter->first;
    m_allocated_requests.erase(request_iter);
  }
  //! Checks that the request is in the list
  Iterator findRequest(MPI_Request request)
  {
    if (m_no_check_request)
      return m_allocated_requests.end();

    if (_isEmptyRequest(request))
      return m_allocated_requests.end();
    auto ireq = m_allocated_requests.find(request);
    if (ireq == m_allocated_requests.end()) {
      if (m_is_report_error_in_request || m_request_error_is_fatal) {
        error() << "MpiAdapter::testRequest() request not referenced "
                << " id=" << request;
        _checkFatalInRequest();
      }
    }
    return ireq;
  }

 private:

  /*!
   * \warning This function must be called with the mpi_lock lock active.
   */
  void _addRequest(MPI_Request request, const TraceInfo& trace_info)
  {
    if (request == MPI_REQUEST_NULL) {
      if (m_is_report_error_in_request || m_request_error_is_fatal) {
        error() << "MpiAdapter::_addRequest() trying to add null request";
        _checkFatalInRequest();
      }
      return;
    }
    if (_isEmptyRequest(request))
      return;
    ++m_total_added_request;
    //info() << "MPI_ADAPTER:ADD REQUEST " << request;
    auto i = m_allocated_requests.find(request);
    if (i != m_allocated_requests.end()) {
      if (m_is_report_error_in_request || m_request_error_is_fatal) {
        error() << "MpiAdapter::_addRequest() request already referenced "
                << " id=" << request;
        _checkFatalInRequest();
      }
      return;
    }
    RequestInfo rinfo;
    rinfo.m_trace = trace_info;
    if (m_use_trace_full_stack)
      rinfo.m_stack_trace = Platform::getStackTrace();
    m_allocated_requests.insert(std::make_pair(request, rinfo));
  }

  /*!
   * \warning This function must be called with the mpi_lock lock active.
   */
  void _removeRequest(MPI_Request request)
  {
    //info() << "MPI_ADAPTER:REMOVE REQUEST " << request;
    if (request == MPI_REQUEST_NULL) {
      if (m_is_report_error_in_request || m_request_error_is_fatal) {
        error() << "MpiAdapter::_removeRequest() null request (" << MPI_REQUEST_NULL << ")";
        _checkFatalInRequest();
      }
      return;
    }
    if (_isEmptyRequest(request))
      return;
    auto i = m_allocated_requests.find(request);
    if (i == m_allocated_requests.end()) {
      if (m_is_report_error_in_request || m_request_error_is_fatal) {
        error() << "MpiAdapter::_removeRequest() request not referenced "
                << " id=" << request;
        _checkFatalInRequest();
      }
    }
    else
      m_allocated_requests.erase(i);
  }

 public:

  void _checkFatalInRequest()
  {
    if (m_request_error_is_fatal)
      ARCCORE_FATAL("Error in requests management");
  }
  Int64 nbRequest() const { return m_allocated_requests.size(); }
  Int64 totalAddedRequest() const { return m_total_added_request; }
  void printRequests() const
  {
    info() << "PRINT REQUESTS\n";
    for (auto& x : m_allocated_requests) {
      info() << "Request id=" << x.first << " trace=" << x.second.m_trace
             << " stack=" << x.second.m_stack_trace;
    }
  }
  void setEmptyRequests(MPI_Request r1, MPI_Request r2)
  {
    m_empty_request1 = r1;
    m_empty_request2 = r2;
  }

 public:

  bool m_request_error_is_fatal = false;
  bool m_is_report_error_in_request = true;
  bool m_trace_mpirequest = false;
  //! True if requests are not checked
  bool m_no_check_request = true;

 private:

  std::map<MPI_Request, RequestInfo> m_allocated_requests;
  bool m_use_trace_full_stack = false;
  MPI_Request m_empty_request1 = MPI_REQUEST_NULL;
  MPI_Request m_empty_request2 = MPI_REQUEST_NULL;
  Int64 m_total_added_request = 0;
  Ref<ITraceMng> m_trace_mng_ref;

 private:

  bool _isEmptyRequest(MPI_Request r) const
  {
    return (r == m_empty_request1 || r == m_empty_request2);
  }
};

#define ARCCORE_ADD_REQUEST(request) \
  m_request_set->addRequest(request, A_FUNCINFO);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  int _checkSize(Int64 i64_size)
  {
    if (i64_size > INT32_MAX)
      ARCCORE_FATAL("Can not convert '{0}' to type integer", i64_size);
    return (int)i64_size;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiAdapter::
MpiAdapter(ITraceMng* trace, IStat* stat, MPI_Comm comm,
           MpiLock* mpi_lock, IMpiProfiling* mpi_op)
: TraceAccessor(trace)
, m_stat(stat)
, m_mpi_lock(mpi_lock)
, m_mpi_prof(mpi_op)
, m_communicator(comm)
, m_comm_rank(0)
, m_comm_size(0)
, m_empty_request1(MPI_REQUEST_NULL)
, m_empty_request2(MPI_REQUEST_NULL)
{
  m_request_set = new RequestSet(trace);

  if (Platform::getEnvironmentVariable("ARCCORE_TRACE_MPI") == "TRUE")
    m_is_trace = true;
  {
    String s = Platform::getEnvironmentVariable("ARCCORE_ALLOW_NULL_RANK_FOR_MPI_ANY_SOURCE");
    if (s == "1" || s == "TRUE")
      m_is_allow_null_rank_for_any_source = true;
    if (s == "0" || s == "FALSE")
      m_is_allow_null_rank_for_any_source = false;
  }

  ::MPI_Comm_rank(m_communicator, &m_comm_rank);
  ::MPI_Comm_size(m_communicator, &m_comm_size);

  // By default, we do not do MPI profiling; we will use the appropriate set
  // method to change it
  if (!m_mpi_prof)
    m_mpi_prof = new NoMpiProfiling();

  /*!
   * This type of request is used by openmpi starting from version 1.8 (it needs
   * to be checked for 1.6, knowing that 1.4 and 1.5 do not have it).
   * This request works somewhat like MPI_REQUEST_NULL and it is possible that
   * it is returned multiple times. Therefore, this request should not be put
   * into m_allocated_requests. We cannot directly access the address of this
   * empty request, but the 1.8 implementation of openmpi returns this request
   * when calling an IRecv with an MPI_PROC_NULL source. We therefore retrieve
   * the value like this.
   */
  MPI_Irecv(m_recv_buffer_for_empty_request, 1, MPI_CHAR, MPI_PROC_NULL,
            50505, m_communicator, &m_empty_request1);

  /*
   * Starting from version 4 of openmpi, it also seems that sends with small
   * buffers always generate the same request. Therefore, it must also be
   * removed from the requests to test. We also post the corresponding
   * MPI_Recv to prevent MPI_ISend from being unintentionally used in a
   * user MPI_Recv (e.g., via MPI_Recv(MPI_ANY_TAG)).
   */
  m_send_buffer_for_empty_request2[0] = 0;
  MPI_Isend(m_send_buffer_for_empty_request2, 1, MPI_CHAR, m_comm_rank,
            50505, m_communicator, &m_empty_request2);

  MPI_Recv(m_recv_buffer_for_empty_request2, 1, MPI_CHAR, m_comm_rank,
           50505, m_communicator, MPI_STATUS_IGNORE);

  m_request_set->setEmptyRequests(m_empty_request1, m_empty_request2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiAdapter::
~MpiAdapter()
{
  if (m_empty_request1 != MPI_REQUEST_NULL)
    MPI_Request_free(&m_empty_request1);
  if (m_empty_request2 != MPI_REQUEST_NULL)
    MPI_Request_free(&m_empty_request2);

  delete m_request_set;
  delete m_mpi_prof;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
buildRequest(int ret, MPI_Request mpi_request)
{
  return MpiRequest(ret, this, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
_checkHasNoRequests()
{
  Int64 nb_request = m_request_set->nbRequest();
  // We cannot perform this test in the destructor because it could
  // potentially throw an exception, and this should not be done in a destructor.
  if (nb_request != 0) {
    warning() << " Pending mpi requests size=" << nb_request;
    m_request_set->printRequests();
    _checkFatalInRequest();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
destroy()
{
  _checkHasNoRequests();
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
setRequestErrorAreFatal(bool v)
{
  m_request_set->m_request_error_is_fatal = v;
}
bool MpiAdapter::
isRequestErrorAreFatal() const
{
  return m_request_set->m_request_error_is_fatal;
}

void MpiAdapter::
setPrintRequestError(bool v)
{
  m_request_set->m_is_report_error_in_request = v;
}
bool MpiAdapter::
isPrintRequestError() const
{
  return m_request_set->m_is_report_error_in_request;
}

void MpiAdapter::
setCheckRequest(bool v)
{
  m_request_set->m_no_check_request = !v;
}

bool MpiAdapter::
isCheckRequest() const
{
  return !m_request_set->m_no_check_request;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
toMPISize(Int64 count)
{
  return _checkSize(count);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
_trace(const char* function)
{
  if (m_is_trace) {
    IStackTraceService* stack_service = Platform::getStackTraceService();
    if (stack_service)
      info() << "MPI_TRACE: " << function << "\n"
             << stack_service->stackTrace().toString();
    else
      info() << "MPI_TRACE: " << function;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
broadcast(void* buf, Int64 nb_elem, Int32 root, MPI_Datatype datatype)
{
  int _nb_elem = _checkSize(nb_elem);
  _trace(MpiInfo(eMpiName::Bcast).name().localstr());
  double begin_time = MPI_Wtime();
  if (m_is_trace)
    info() << "MPI_TRACE: MPI broadcast: before"
           << " buf=" << buf
           << " nb_elem=" << nb_elem
           << " root=" << root
           << " datatype=" << datatype;

  m_mpi_prof->broadcast(buf, _nb_elem, datatype, root, m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Bcast).name(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingBroadcast(void* buf, Int64 nb_elem, Int32 root, MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
  int _nb_elem = _checkSize(nb_elem);
  _trace(" MPI_Bcast");
  double begin_time = MPI_Wtime();
  ret = MPI_Ibcast(buf, _nb_elem, datatype, root, m_communicator, &mpi_request);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add("IBroadcast", sr_time, 0);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
gather(const void* send_buf, void* recv_buf, Int64 nb_elem, Int32 root, MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = _checkSize(nb_elem);
  int _root = static_cast<int>(root);
  _trace(MpiInfo(eMpiName::Gather).name().localstr());
  double begin_time = MPI_Wtime();
  m_mpi_prof->gather(_sbuf, _nb_elem, datatype, recv_buf, _nb_elem, datatype, _root, m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Gather).name(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingGather(const void* send_buf, void* recv_buf,
                  Int64 nb_elem, Int32 root, MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = _checkSize(nb_elem);
  int _root = static_cast<int>(root);
  _trace("MPI_Igather");
  double begin_time = MPI_Wtime();
  ret = MPI_Igather(_sbuf, _nb_elem, datatype, recv_buf, _nb_elem, datatype, _root,
                    m_communicator, &mpi_request);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add("IGather", sr_time, 0);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
allGather(const void* send_buf, void* recv_buf,
          Int64 nb_elem, MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = _checkSize(nb_elem);
  _trace(MpiInfo(eMpiName::Allgather).name().localstr());
  double begin_time = MPI_Wtime();
  m_mpi_prof->allGather(_sbuf, _nb_elem, datatype, recv_buf, _nb_elem, datatype, m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Allgather).name(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllGather(const void* send_buf, void* recv_buf,
                     Int64 nb_elem, MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = _checkSize(nb_elem);
  _trace("MPI_Iallgather");
  double begin_time = MPI_Wtime();
  ret = MPI_Iallgather(_sbuf, _nb_elem, datatype, recv_buf, _nb_elem, datatype,
                       m_communicator, &mpi_request);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add("IAllGather", sr_time, 0);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
gatherVariable(const void* send_buf, void* recv_buf, const int* recv_counts,
               const int* recv_indexes, Int64 nb_elem, Int32 root, MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = _checkSize(nb_elem);
  int _root = static_cast<int>(root);
  _trace(MpiInfo(eMpiName::Gatherv).name().localstr());
  double begin_time = MPI_Wtime();
  m_mpi_prof->gatherVariable(_sbuf, _nb_elem, datatype, recv_buf, recv_counts, recv_indexes, datatype, _root, m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Gatherv).name().localstr(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
allGatherVariable(const void* send_buf, void* recv_buf, const int* recv_counts,
                  const int* recv_indexes, Int64 nb_elem, MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = _checkSize(nb_elem);
  _trace(MpiInfo(eMpiName::Allgatherv).name().localstr());
  //info() << " ALLGATHERV N=" << _nb_elem;
  //for( int i=0; i<m_comm_size; ++i )
  //info() << " ALLGATHERV I=" << i << " recv_count=" << recv_counts[i]
  //     << " recv_indexes=" << recv_indexes[i];
  double begin_time = MPI_Wtime();
  m_mpi_prof->allGatherVariable(_sbuf, _nb_elem, datatype, recv_buf, recv_counts, recv_indexes, datatype, m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Allgatherv).name().localstr(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
scatterVariable(const void* send_buf, const int* send_count, const int* send_indexes,
                void* recv_buf, Int64 nb_elem, Int32 root, MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int* _send_count = const_cast<int*>(send_count);
  int* _send_indexes = const_cast<int*>(send_indexes);
  int _nb_elem = _checkSize(nb_elem);
  _trace(MpiInfo(eMpiName::Scatterv).name().localstr());
  double begin_time = MPI_Wtime();
  m_mpi_prof->scatterVariable(_sbuf,
                              _send_count,
                              _send_indexes,
                              datatype,
                              recv_buf,
                              _nb_elem,
                              datatype,
                              root,
                              m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Scatterv).name(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
allToAll(const void* send_buf, void* recv_buf, Integer count, MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int icount = _checkSize(count);
  _trace(MpiInfo(eMpiName::Alltoall).name().localstr());
  double begin_time = MPI_Wtime();
  m_mpi_prof->allToAll(_sbuf, icount, datatype, recv_buf, icount, datatype, m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Alltoall).name().localstr(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllToAll(const void* send_buf, void* recv_buf, Integer count, MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
  void* _sbuf = const_cast<void*>(send_buf);
  int icount = _checkSize(count);
  _trace("MPI_IAlltoall");
  double begin_time = MPI_Wtime();
  ret = MPI_Ialltoall(_sbuf, icount, datatype, recv_buf, icount, datatype, m_communicator, &mpi_request);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add("IAllToAll", sr_time, 0);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
allToAllVariable(const void* send_buf, const int* send_counts,
                 const int* send_indexes, void* recv_buf, const int* recv_counts,
                 const int* recv_indexes, MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int* _send_counts = const_cast<int*>(send_counts);
  int* _send_indexes = const_cast<int*>(send_indexes);
  int* _recv_counts = const_cast<int*>(recv_counts);
  int* _recv_indexes = const_cast<int*>(recv_indexes);

  _trace(MpiInfo(eMpiName::Alltoallv).name().localstr());
  double begin_time = MPI_Wtime();
  m_mpi_prof->allToAllVariable(_sbuf, _send_counts, _send_indexes, datatype,
                               recv_buf, _recv_counts, _recv_indexes, datatype, m_communicator);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add(MpiInfo(eMpiName::Alltoallv).name(), sr_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllToAllVariable(const void* send_buf, const int* send_counts,
                            const int* send_indexes, void* recv_buf, const int* recv_counts,
                            const int* recv_indexes, MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
  void* _sbuf = const_cast<void*>(send_buf);
  int* _send_counts = const_cast<int*>(send_counts);
  int* _send_indexes = const_cast<int*>(send_indexes);
  int* _recv_counts = const_cast<int*>(recv_counts);
  int* _recv_indexes = const_cast<int*>(recv_indexes);

  _trace("MPI_Ialltoallv");
  double begin_time = MPI_Wtime();
  ret = MPI_Ialltoallv(_sbuf, _send_counts, _send_indexes, datatype,
                       recv_buf, _recv_counts, _recv_indexes, datatype,
                       m_communicator, &mpi_request);
  double end_time = MPI_Wtime();
  double sr_time = (end_time - begin_time);
  //TODO determine the message size
  m_stat->add("IAllToAll", sr_time, 0);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
barrier()
{
  // TODO: theoretically there should not be any pending requests
  // between two barriers to avoid any issues.
  // _checkHasNoRequests();
  // TODO add corresponding trace for profiling.
  MPI_Barrier(m_communicator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingBarrier()
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
  ret = MPI_Ibarrier(m_communicator, &mpi_request);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
allReduce(const void* send_buf, void* recv_buf, Int64 count, MPI_Datatype datatype, MPI_Op op)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = _checkSize(count);
  double begin_time = MPI_Wtime();
  _trace(MpiInfo(eMpiName::Allreduce).name().localstr());
  try {
    ++m_nb_all_reduce;
    m_mpi_prof->allReduce(_sbuf, recv_buf, _n, datatype, op, m_communicator);
  }
  catch (TimeoutException& ex) {
    std::ostringstream ostr;
    ostr << "MPI_Allreduce"
         << " send_buf=" << send_buf
         << " recv_buf=" << recv_buf
         << " n=" << count
         << " datatype=" << datatype
         << " op=" << op
         << " NB=" << m_nb_all_reduce;
    ex.setAdditionalInfo(ostr.str());
    throw;
  }
  double end_time = MPI_Wtime();
  m_stat->add(MpiInfo(eMpiName::Allreduce).name(), end_time - begin_time, count);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllReduce(const void* send_buf, void* recv_buf, Int64 count, MPI_Datatype datatype, MPI_Op op)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = _checkSize(count);
  double begin_time = MPI_Wtime();
  _trace("MPI_IAllreduce");
  ret = MPI_Iallreduce(_sbuf, recv_buf, _n, datatype, op, m_communicator, &mpi_request);
  double end_time = MPI_Wtime();
  m_stat->add("IReduce", end_time - begin_time, _n);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
reduce(const void* send_buf, void* recv_buf, Int64 count, MPI_Datatype datatype, MPI_Op op, Integer root)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = _checkSize(count);
  int _root = static_cast<int>(root);
  double begin_time = MPI_Wtime();
  _trace(MpiInfo(eMpiName::Reduce).name().localstr());
  try {
    ++m_nb_reduce;
    m_mpi_prof->reduce(_sbuf, recv_buf, _n, datatype, op, _root, m_communicator);
  }
  catch (TimeoutException& ex) {
    std::ostringstream ostr;
    ostr << "MPI_reduce"
         << " send_buf=" << send_buf
         << " recv_buf=" << recv_buf
         << " n=" << count
         << " datatype=" << datatype
         << " op=" << op
         << " root=" << root
         << " NB=" << m_nb_reduce;
    ex.setAdditionalInfo(ostr.str());
    throw;
  }

  double end_time = MPI_Wtime();
  m_stat->add(MpiInfo(eMpiName::Reduce).name(), end_time - begin_time, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
scan(const void* send_buf, void* recv_buf, Int64 count, MPI_Datatype datatype, MPI_Op op)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = _checkSize(count);
  double begin_time = MPI_Wtime();
  _trace(MpiInfo(eMpiName::Scan).name().localstr());
  m_mpi_prof->scan(_sbuf, recv_buf, _n, datatype, op, m_communicator);
  double end_time = MPI_Wtime();
  m_stat->add(MpiInfo(eMpiName::Scan).name(), end_time - begin_time, count);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
directSendRecv(const void* send_buffer, Int64 send_buffer_size,
               void* recv_buffer, Int64 recv_buffer_size,
               Int32 proc, Int64 elem_size, MPI_Datatype data_type)
{
  void* v_send_buffer = const_cast<void*>(send_buffer);
  MPI_Status mpi_status;
  double begin_time = MPI_Wtime();
  _trace(MpiInfo(eMpiName::Sendrecv).name().localstr());
  int sbuf_size = _checkSize(send_buffer_size);
  int rbuf_size = _checkSize(recv_buffer_size);
  m_mpi_prof->sendRecv(v_send_buffer, sbuf_size, data_type, proc, 99,
                       recv_buffer, rbuf_size, data_type, proc, 99,
                       m_communicator, &mpi_status);
  double end_time = MPI_Wtime();
  Int64 send_size = send_buffer_size * elem_size;
  Int64 recv_size = recv_buffer_size * elem_size;
  double sr_time = (end_time - begin_time);

  //debug(Trace::High) << "MPI SendRecv: send " << send_size << " recv "
  //                      << recv_size << " time " << sr_time ;
  m_stat->add(MpiInfo(eMpiName::Sendrecv).name(), sr_time, send_size + recv_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
sendNonBlockingNoStat(const void* send_buffer, Int64 send_buffer_size,
                      Int32 dest_rank, MPI_Datatype data_type, int mpi_tag)
{
  void* v_send_buffer = const_cast<void*>(send_buffer);
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int sbuf_size = _checkSize(send_buffer_size);
  int ret = 0;
  m_mpi_prof->iSend(v_send_buffer, sbuf_size, data_type, dest_rank, mpi_tag, m_communicator, &mpi_request);
  if (m_is_trace)
    info() << " ISend ret=" << ret << " proc=" << dest_rank << " tag=" << mpi_tag << " request=" << mpi_request;
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directSend(const void* send_buffer, Int64 send_buffer_size,
           Int32 proc, Int64 elem_size, MPI_Datatype data_type,
           int mpi_tag, bool is_blocked)
{
  void* v_send_buffer = const_cast<void*>(send_buffer);
  MPI_Request mpi_request = MPI_REQUEST_NULL;

  double begin_time = 0.0;
  double end_time = 0.0;
  Int64 send_size = send_buffer_size * elem_size;
  int ret = 0;
  if (m_is_trace)
    info() << "MPI_TRACE: MPI Send: send before"
           << " size=" << send_size
           << " dest=" << proc
           << " tag=" << mpi_tag
           << " datatype=" << data_type
           << " blocking " << is_blocked;
  if (is_blocked) {
    // if m_mpi_lock is not null, we must
    // use an MPI_ISend followed by an
    // active MPI_Test loop to avoid any
    // dead lock issues.
    if (m_mpi_lock) {
      {
        MpiLock::Section mls(m_mpi_lock);
        begin_time = MPI_Wtime();
        int sbuf_size = _checkSize(send_buffer_size);
        m_mpi_prof->iSend(v_send_buffer, sbuf_size, data_type, proc, mpi_tag, m_communicator, &mpi_request);
      }
      int is_finished = 0;
      MPI_Status mpi_status;
      while (is_finished == 0) {
        MpiLock::Section mls(m_mpi_lock);
        MPI_Request_get_status(mpi_request, &is_finished, &mpi_status);
        if (is_finished != 0) {
          m_mpi_prof->wait(&mpi_request, (MPI_Status*)MPI_STATUS_IGNORE);
          end_time = MPI_Wtime();
          mpi_request = MPI_REQUEST_NULL;
        }
      }
    }
    else {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      int sbuf_size = _checkSize(send_buffer_size);
      m_mpi_prof->send(v_send_buffer, sbuf_size, data_type, proc, mpi_tag, m_communicator);
      end_time = MPI_Wtime();
    }
  }
  else {
    {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      int sbuf_size = _checkSize(send_buffer_size);
      m_mpi_prof->iSend(v_send_buffer, sbuf_size, data_type, proc, mpi_tag, m_communicator, &mpi_request);
      if (m_is_trace)
        info() << " ISend ret=" << ret << " proc=" << proc << " tag=" << mpi_tag << " request=" << mpi_request;
      end_time = MPI_Wtime();
      ARCCORE_ADD_REQUEST(mpi_request);
    }
    if (m_is_trace) {
      info() << "MPI Send: send after"
             << " request=" << mpi_request;
    }
  }
  double sr_time = (end_time - begin_time);

  debug(Trace::High) << "MPI Send: send " << send_size
                     << " time " << sr_time << " blocking " << is_blocked;
  // TODO(FL): look into how to profile Isend
  m_stat->add(MpiInfo(eMpiName::Send).name(), end_time - begin_time, send_size);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directSendPack(const void* send_buffer, Int64 send_buffer_size,
               Int32 proc, int mpi_tag, bool is_blocked)
{
  return directSend(send_buffer, send_buffer_size, proc, 1, MPI_PACKED, mpi_tag, is_blocked);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMessagePassingMng* MpiAdapter::
commSplit(bool keep)
{
  MPI_Comm new_comm;

  MPI_Comm_split(m_communicator, (keep) ? 1 : MPI_UNDEFINED, commRank(), &new_comm);
  if (keep) {
    // Failed if new_comm is MPI_COMM_NULL
    return StandaloneMpiMessagePassingMng::create(new_comm, true);
  }
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
receiveNonBlockingNoStat(void* recv_buffer, Int64 recv_buffer_size,
                         Int32 source_rank, MPI_Datatype data_type, int mpi_tag)
{
  int rbuf_size = _checkSize(recv_buffer_size);
  int ret = 0;
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  m_mpi_prof->iRecv(recv_buffer, rbuf_size, data_type, source_rank, mpi_tag, m_communicator, &mpi_request);
  ARCCORE_ADD_REQUEST(mpi_request);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directRecv(void* recv_buffer, Int64 recv_buffer_size,
           Int32 proc, Int64 elem_size, MPI_Datatype data_type,
           int mpi_tag, bool is_blocked)
{
  MPI_Status mpi_status;
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = 0;
  double begin_time = 0.0;
  double end_time = 0.0;

  int i_proc = 0;
  if (proc == A_PROC_NULL_RANK)
    ARCCORE_THROW(NotImplementedException, "Receive with MPI_PROC_NULL");
  if (proc == A_NULL_RANK && !m_is_allow_null_rank_for_any_source)
    ARCCORE_FATAL("Can not use A_NULL_RANK for any source. Use A_ANY_SOURCE_RANK instead");
  if (proc == A_NULL_RANK || proc == A_ANY_SOURCE_RANK)
    i_proc = MPI_ANY_SOURCE;
  else
    i_proc = static_cast<int>(proc);

  Int64 recv_size = recv_buffer_size * elem_size;
  if (m_is_trace) {
    info() << "MPI_TRACE: MPI Recv: recv before "
           << " size=" << recv_size
           << " from=" << i_proc
           << " tag=" << mpi_tag
           << " datatype=" << data_type
           << " blocking=" << is_blocked;
  }
  if (is_blocked) {
    // if m_mpi_lock is not null, we must
    // use an MPI_IRecv followed by an
    // active MPI_Test loop to avoid any
    // dead lock issues.
    if (m_mpi_lock) {
      {
        MpiLock::Section mls(m_mpi_lock);
        begin_time = MPI_Wtime();
        int rbuf_size = _checkSize(recv_buffer_size);
        m_mpi_prof->iRecv(recv_buffer, rbuf_size, data_type, i_proc, mpi_tag, m_communicator, &mpi_request);
      }
      int is_finished = 0;
      MPI_Status mpi_status;
      while (is_finished == 0) {
        MpiLock::Section mls(m_mpi_lock);
        MPI_Request_get_status(mpi_request, &is_finished, &mpi_status);
        if (is_finished != 0) {
          end_time = MPI_Wtime();
          m_mpi_prof->wait(&mpi_request, (MPI_Status*)MPI_STATUS_IGNORE);
          mpi_request = MPI_REQUEST_NULL;
        }
      }
    }
    else {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      int rbuf_size = _checkSize(recv_buffer_size);
      m_mpi_prof->recv(recv_buffer, rbuf_size, data_type, i_proc, mpi_tag, m_communicator, &mpi_status);
      end_time = MPI_Wtime();
    }
  }
  else {
    {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      int rbuf_size = _checkSize(recv_buffer_size);
      m_mpi_prof->iRecv(recv_buffer, rbuf_size, data_type, i_proc, mpi_tag, m_communicator, &mpi_request);
      end_time = MPI_Wtime();
      ARCCORE_ADD_REQUEST(mpi_request);
    }
    if (m_is_trace) {
      info() << "MPI Recv: recv after "
             << " request=" << mpi_request;
    }
  }
  double sr_time = (end_time - begin_time);

  debug(Trace::High) << "MPI Recv: recv after " << recv_size
                     << " time " << sr_time << " blocking " << is_blocked;
  m_stat->add(MpiInfo(eMpiName::Recv).name(), end_time - begin_time, recv_size);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
probeRecvPack(UniqueArray<Byte>& recv_buffer, Int32 proc)
{
  double begin_time = MPI_Wtime();
  MPI_Status status;
  int recv_buffer_size = 0;
  _trace("MPI_Probe");
  m_mpi_prof->probe(proc, 101, m_communicator, &status);
  m_mpi_prof->getCount(&status, MPI_PACKED, &recv_buffer_size);

  recv_buffer.resize(recv_buffer_size);
  m_mpi_prof->recv(recv_buffer.data(), recv_buffer_size, MPI_PACKED, proc, 101, m_communicator, &status);

  double end_time = MPI_Wtime();
  Int64 recv_size = recv_buffer_size;
  double sr_time = (end_time - begin_time);
  debug(Trace::High) << "MPI probeRecvPack " << recv_size
                     << " time " << sr_time;
  m_stat->add(MpiInfo(eMpiName::Recv).name(), end_time - begin_time, recv_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo MpiAdapter::
_buildSourceInfoFromStatus(const MPI_Status& mpi_status)
{
  // Retrieves the message size in bytes.
  MPI_Count message_size = 0;
  MPI_Get_elements_x(&mpi_status, MPI_BYTE, &message_size);
  MessageTag tag(mpi_status.MPI_TAG);
  MessageRank rank(mpi_status.MPI_SOURCE);
  return MessageSourceInfo(rank, tag, message_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageId MpiAdapter::
_probeMessage(MessageRank source, MessageTag tag, bool is_blocking)
{
  MPI_Status mpi_status;
  int has_message = 0;
  MPI_Message message;
  int ret = 0;
  int mpi_source = source.value();
  if (source.isProcNull())
    ARCCORE_THROW(NotImplementedException, "Probe with MPI_PROC_NULL");
  if (source.isNull() && !m_is_allow_null_rank_for_any_source)
    ARCCORE_FATAL("Can not use MPI_Mprobe with null rank. Use MessageRank::anySourceRank() instead");
  if (source.isNull() || source.isAnySource())
    mpi_source = MPI_ANY_SOURCE;
  int mpi_tag = tag.value();
  if (tag.isNull())
    mpi_tag = MPI_ANY_TAG;
  if (is_blocking) {
    ret = MPI_Mprobe(mpi_source, mpi_tag, m_communicator, &message, &mpi_status);
    has_message = true;
  }
  else {
    ret = MPI_Improbe(mpi_source, mpi_tag, m_communicator, &has_message, &message, &mpi_status);
  }
  if (ret != 0)
    ARCCORE_FATAL("Error during call to MPI_Mprobe r={0}", ret);
  MessageId ret_message;
  if (has_message != 0) {
    MessageSourceInfo si(_buildSourceInfoFromStatus(mpi_status));
    ret_message = MessageId(si, message);
  }
  return ret_message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageId MpiAdapter::
probeMessage(PointToPointMessageInfo message)
{
  if (!message.isValid())
    return MessageId();

  // The message must be initialized with a (rank/tag) pair.
  if (!message.isRankTag())
    ARCCORE_FATAL("Invalid message_info: message.isRankTag() is false");

  return _probeMessage(message.destinationRank(), message.tag(), message.isBlocking());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo MpiAdapter::
_legacyProbeMessage(MessageRank source, MessageTag tag, bool is_blocking)
{
  MPI_Status mpi_status;
  int has_message = 0;
  int ret = 0;
  int mpi_source = source.value();
  if (source.isProcNull())
    ARCCORE_THROW(NotImplementedException, "Probe with MPI_PROC_NULL");
  if (source.isNull() && !m_is_allow_null_rank_for_any_source)
    ARCCORE_FATAL("Can not use MPI_Probe with null rank. Use MessageRank::anySourceRank() instead");
  if (source.isNull() || source.isAnySource())
    mpi_source = MPI_ANY_SOURCE;
  int mpi_tag = tag.value();
  if (tag.isNull())
    mpi_tag = MPI_ANY_TAG;
  if (is_blocking) {
    ret = MPI_Probe(mpi_source, mpi_tag, m_communicator, &mpi_status);
    has_message = true;
  }
  else
    ret = MPI_Iprobe(mpi_source, mpi_tag, m_communicator, &has_message, &mpi_status);
  if (ret != 0)
    ARCCORE_FATAL("Error during call to MPI_Mprobe r={0}", ret);
  if (has_message != 0)
    return _buildSourceInfoFromStatus(mpi_status);
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo MpiAdapter::
legacyProbeMessage(PointToPointMessageInfo message)
{
  if (!message.isValid())
    return {};

  // The message must be initialized with a (rank/tag) pair.
  if (!message.isRankTag())
    ARCCORE_FATAL("Invalid message_info: message.isRankTag() is false");

  return _legacyProbeMessage(message.destinationRank(), message.tag(), message.isBlocking());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Reception via MPI_Mrecv() or MPI_Imrecv()
Request MpiAdapter::
directRecv(void* recv_buffer, Int64 recv_buffer_size,
           MessageId message, Int64 elem_size, MPI_Datatype data_type,
           bool is_blocked)
{
  MPI_Status mpi_status;
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  MPI_Message mpi_message = (MPI_Message)message;
  int ret = 0;
  double begin_time = 0.0;
  double end_time = 0.0;

  Int64 recv_size = recv_buffer_size * elem_size;
  if (m_is_trace) {
    info() << "MPI_TRACE: MPI Mrecv: recv before "
           << " size=" << recv_size
           << " from_msg=" << message
           << " datatype=" << data_type
           << " blocking=" << is_blocked;
  }
  if (is_blocked) {
    // if m_mpi_lock is not null, we must
    // use an MPI_IRecv followed by an
    // active MPI_Test loop to avoid any
    // dead lock issues.
    if (m_mpi_lock) {
      {
        MpiLock::Section mls(m_mpi_lock);
        begin_time = MPI_Wtime();
        int rbuf_size = _checkSize(recv_buffer_size);
        MPI_Imrecv(recv_buffer, rbuf_size, data_type, &mpi_message, &mpi_request);
        //m_mpi_prof->iRecv(recv_buffer, rbuf_size, data_type, i_proc, mpi_tag, m_communicator, &mpi_request);
      }
      int is_finished = 0;
      MPI_Status mpi_status;
      while (is_finished == 0) {
        MpiLock::Section mls(m_mpi_lock);
        MPI_Request_get_status(mpi_request, &is_finished, &mpi_status);
        if (is_finished != 0) {
          end_time = MPI_Wtime();
          m_mpi_prof->wait(&mpi_request, (MPI_Status*)MPI_STATUS_IGNORE);
          mpi_request = MPI_REQUEST_NULL;
        }
      }
    }
    else {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      int rbuf_size = _checkSize(recv_buffer_size);
      MPI_Mrecv(recv_buffer, rbuf_size, data_type, &mpi_message, &mpi_status);
      //m_mpi_prof->recv(recv_buffer, rbuf_size, data_type, i_proc, mpi_tag, m_communicator, &mpi_status);
      end_time = MPI_Wtime();
    }
  }
  else {
    {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      int rbuf_size = _checkSize(recv_buffer_size);
      //m_mpi_prof->iRecv(recv_buffer, rbuf_size, data_type, i_proc, mpi_tag, m_communicator, &mpi_request);
      ret = MPI_Imrecv(recv_buffer, rbuf_size, data_type, &mpi_message, &mpi_request);
      //m_mpi_prof->iRecv(recv_buffer, rbuf_size, data_type, i_proc, mpi_tag, m_communicator, &mpi_request);
      end_time = MPI_Wtime();
      ARCCORE_ADD_REQUEST(mpi_request);
    }
    if (m_is_trace) {
      info() << "MPI Recv: recv after "
             << " request=" << mpi_request;
    }
  }
  double sr_time = (end_time - begin_time);

  debug(Trace::High) << "MPI Recv: recv after " << recv_size
                     << " time " << sr_time << " blocking " << is_blocked;
  m_stat->add(MpiInfo(eMpiName::Recv).name(), end_time - begin_time, recv_size);
  return buildRequest(ret, mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directRecvPack(void* recv_buffer, Int64 recv_buffer_size,
               Int32 proc, int mpi_tag, bool is_blocking)
{
  return directRecv(recv_buffer, recv_buffer_size, proc, 1, MPI_PACKED, mpi_tag, is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// FIXME: Implement direct method with MPI_STATUS_IGNORE
void MpiAdapter::
waitAllRequests(ArrayView<Request> requests)
{
  UniqueArray<bool> indexes(requests.size());
  UniqueArray<MPI_Status> mpi_status(requests.size());
  while (_waitAllRequestsMPI(requests, indexes, mpi_status)) {
    ; // Continue as long as there are requests.
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// FIXME: Implement direct method with MPI_STATUS_IGNORE
void MpiAdapter::
waitSomeRequests(ArrayView<Request> requests,
                 ArrayView<bool> indexes,
                 bool is_non_blocking)
{
  UniqueArray<MPI_Status> mpi_status(requests.size());
  waitSomeRequestsMPI(requests, indexes, mpi_status, is_non_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct MpiAdapter::SubRequestInfo
{
  SubRequestInfo(Ref<ISubRequest> sr, Integer i, int source_rank, int source_tag)
  : sub_request(sr)
  , index(i)
  , mpi_source_rank(source_rank)
  , mpi_source_tag(source_tag)
  {}

  Ref<ISubRequest> sub_request;
  Integer index = -1;
  int mpi_source_rank = MPI_PROC_NULL;
  int mpi_source_tag = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MpiAdapter::
_handleEndRequests(ArrayView<Request> requests, ArrayView<bool> done_indexes,
                   ArrayView<MPI_Status> status)
{
  UniqueArray<SubRequestInfo> new_requests;
  Integer size = requests.size();
  {
    MpiLock::Section mls(m_mpi_lock);
    for (Integer i = 0; i < size; ++i) {
      if (done_indexes[i]) {
        // Be careful to use a reference, otherwise the reset won't
        // apply to the correct variable
        Request& r = requests[i];
        // Note: the request might not be valid (for example, if it is)
        // a blocking request but still have a sub-request.
        if (r.hasSubRequest()) {
          if (m_is_trace)
            info() << "Done request with sub-request r=" << r << " mpi_r=" << r << " i=" << i
                   << " source_rank=" << status[i].MPI_SOURCE
                   << " source_tag=" << status[i].MPI_TAG;
          new_requests.add(SubRequestInfo(r.subRequest(), i, status[i].MPI_SOURCE, status[i].MPI_TAG));
        }
        if (r.isValid()) {
          _removeRequest((MPI_Request)(r));
          r.reset();
        }
      }
    }
  }

  // NOTE: calls to sub-requests can generate other requests.
  // Care must be taken not to use sub-requests while the lock is active.
  bool has_new_request = false;
  if (!new_requests.empty()) {
    // Contains the status of the i-th request
    UniqueArray<MPI_Status> old_status(size);
    {
      Integer index = 0;
      for (Integer i = 0; i < size; ++i) {
        if (done_indexes[i]) {
          old_status[i] = status[index];
          ++index;
        }
      }
    }
    // If there are new requests, the values in 'status' must be shifted
    for (SubRequestInfo& sri : new_requests) {
      Integer index = sri.index;
      if (m_is_trace)
        info() << "Before handle new request index=" << index
               << " sri.source_rank=" << sri.mpi_source_rank
               << " sri.source_tag=" << sri.mpi_source_tag;
      SubRequestCompletionInfo completion_info(MessageRank(old_status[index].MPI_SOURCE), MessageTag(old_status[index].MPI_TAG));
      Request r = sri.sub_request->executeOnCompletion(completion_info);
      if (m_is_trace)
        info() << "Handle new request index=" << index << " old_r=" << requests[index] << " new_r=" << r;
      // If there is a new request, it replaces
      // the old one, so we must act as if
      // the original request is not finished.
      if (r.isValid()) {
        has_new_request = true;
        requests[index] = r;
        done_indexes[index] = false;
      }
    }
    {
      Integer index = 0;
      for (Integer i = 0; i < size; ++i) {
        if (done_indexes[i]) {
          status[index] = old_status[i];
          ++index;
        }
      }
    }
  }
  return has_new_request;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MpiAdapter::
_waitAllRequestsMPI(ArrayView<Request> requests,
                    ArrayView<bool> indexes,
                    ArrayView<MPI_Status> mpi_status)
{
  Integer size = requests.size();
  if (size == 0)
    return false;
  //ATTENTION: Mpi modifies this array upon return from MPI_Waitall
  UniqueArray<MPI_Request> mpi_request(size);
  for (Integer i = 0; i < size; ++i) {
    mpi_request[i] = (MPI_Request)(requests[i]);
  }
  if (m_is_trace)
    info() << " MPI_waitall begin size=" << size;
  double diff_time = 0.0;
  if (m_mpi_lock) {
    double begin_time = MPI_Wtime();
    for (Integer i = 0; i < size; ++i) {
      MPI_Request request = (MPI_Request)(mpi_request[i]);
      int is_finished = 0;
      while (is_finished == 0) {
        MpiLock::Section mls(m_mpi_lock);
        m_mpi_prof->test(&request, &is_finished, (MPI_Status*)MPI_STATUS_IGNORE);
      }
    }
    double end_time = MPI_Wtime();
    diff_time = end_time - begin_time;
  }
  else {
    //TODO: transform into a while loop and MPI_Testall if m_mpi_lock is non-null
    MpiLock::Section mls(m_mpi_lock);
    double begin_time = MPI_Wtime();
    m_mpi_prof->waitAll(size, mpi_request.data(), mpi_status.data());
    double end_time = MPI_Wtime();
    diff_time = end_time - begin_time;
  }

  // Indicates that each request has been processed because we performed a waitall.
  for (Integer i = 0; i < size; ++i) {
    indexes[i] = true;
  }

  bool has_new_request = _handleEndRequests(requests, indexes, mpi_status);
  if (m_is_trace)
    info() << " MPI_waitall end size=" << size;
  m_stat->add(MpiInfo(eMpiName::Waitall).name(), diff_time, size);
  return has_new_request;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
waitSomeRequestsMPI(ArrayView<Request> requests, ArrayView<bool> indexes,
                    ArrayView<MPI_Status> mpi_status, bool is_non_blocking)
{
  Integer size = requests.size();
  if (size == 0)
    return;
  //TODO: use StackArray (when they become available...)
  UniqueArray<MPI_Request> mpi_request(size);
  UniqueArray<MPI_Request> saved_mpi_request(size);
  UniqueArray<int> completed_requests(size);
  int nb_completed_request = 0;

  // Save the request to deallocate it in m_allocated_requests,
  // because its value will no longer be valid after calling MPI_Wait*
  for (Integer i = 0; i < size; ++i) {
    // In the case where this method is called multiple times
    // with the same requests array, there may be invalid
    // requests that will crash the MPI call.
    if (!requests[i].isValid()) {
      saved_mpi_request[i] = MPI_REQUEST_NULL;
    }
    else {
      saved_mpi_request[i] = static_cast<MPI_Request>(requests[i]);
    }
  }

  // Only display debug in blocking mode or if explicitly requested to
  // avoid too many messages
  bool is_print_debug = m_is_trace || (!is_non_blocking);
  if (is_print_debug)
    debug() << "WaitRequestBegin is_non_blocking=" << is_non_blocking << " n=" << size;

  double begin_time = MPI_Wtime();

  try {
    if (is_non_blocking) {
      _trace(MpiInfo(eMpiName::Testsome).name().localstr());
      {
        MpiLock::Section mls(m_mpi_lock);
        m_mpi_prof->testSome(size, saved_mpi_request.data(), &nb_completed_request,
                             completed_requests.data(), mpi_status.data());
      }
      //If there is no active handle in the list, it returns outcount = MPI_UNDEFINED.
      if (nb_completed_request == MPI_UNDEFINED) // If no requests were valid.
        nb_completed_request = 0;
      if (is_print_debug)
        debug() << "WaitSomeRequestMPI: TestSome nb_completed=" << nb_completed_request;
    }
    else {
      _trace(MpiInfo(eMpiName::Waitsome).name().localstr());
      {
        // TODO: if the lock exists, a testSome() loop must be performed
        // so as not to block.
        MpiLock::Section mls(m_mpi_lock);
        m_mpi_prof->waitSome(size, saved_mpi_request.data(), &nb_completed_request,
                             completed_requests.data(), mpi_status.data());
      }
      // One must not use mpi_request[i] because it is modified by Mpi
      // mpi_request[i] == MPI_REQUEST_NULL
      if (nb_completed_request == MPI_UNDEFINED) // If no requests were valid.
        nb_completed_request = 0;
      if (is_print_debug)
        debug() << "WaitSomeRequest nb_completed=" << nb_completed_request;
    }
  }
  catch (TimeoutException& ex) {
    std::ostringstream ostr;
    if (is_non_blocking)
      ostr << MpiInfo(eMpiName::Testsome).name();
    else
      ostr << MpiInfo(eMpiName::Waitsome).name();
    ostr << " size=" << size
         << " is_non_blocking=" << is_non_blocking;
    ex.setAdditionalInfo(ostr.str());
    throw;
  }

  for (int z = 0; z < nb_completed_request; ++z) {
    int index = completed_requests[z];
    if (is_print_debug)
      debug() << "Completed my_rank=" << m_comm_rank << " z=" << z
              << " index=" << index
              << " tag=" << mpi_status[z].MPI_TAG
              << " source=" << mpi_status[z].MPI_SOURCE;

    indexes[index] = true;
  }

  bool has_new_request = _handleEndRequests(requests, indexes, mpi_status);
  if (has_new_request) {
    // If there are new requests, it is possible that no
    // request has completed. In the case of testSome, this is not serious.
    // In the case of waitSome, this means that we must wait again.
  }
  double end_time = MPI_Wtime();
  m_stat->add(MpiInfo(eMpiName::Waitsome).name(), end_time - begin_time, size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
freeRequest(Request& request)
{
  if (!request.isValid()) {
    warning() << "MpiAdapter::freeRequest() null request r=" << (MPI_Request)request;
    _checkFatalInRequest();
    return;
  }
  {
    MpiLock::Section mls(m_mpi_lock);

    auto mr = (MPI_Request)request;
    _removeRequest(mr);
    MPI_Request_free(&mr);
  }
  request.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MpiAdapter::
testRequest(Request& request)
{
  // It is allowed by MPI to perform a test with a null request.
  if (!request.isValid())
    return true;

  auto mr = (MPI_Request)request;
  int is_finished = 0;

  {
    MpiLock::Section mls(m_mpi_lock);

    // First, we must retrieve the request location because if it
    // is finished, it will be automatically freed by MPI during the test
    // and thus we will no longer be able to remove it
    RequestSet::Iterator request_iter = m_request_set->findRequest(mr);

    m_mpi_prof->test(&mr, &is_finished, (MPI_Status*)MPI_STATUS_IGNORE);
    //info() << "** TEST REQUEST r=" << mr << " is_finished=" << is_finished;
    if (is_finished != 0) {
      m_request_set->removeRequest(request_iter);
      if (request.hasSubRequest())
        ARCCORE_THROW(NotImplementedException, "SubRequest support");
      request.reset();
      return true;
    }
  }

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \warning This function must be called with the mpi_lock lock active.
 */
void MpiAdapter::
_addRequest(MPI_Request request)
{
  m_request_set->addRequest(request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \warning This function must be called with the mpi_lock lock active.
 */
void MpiAdapter::
_removeRequest(MPI_Request request)
{
  m_request_set->removeRequest(request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
enableDebugRequest(bool enable_debug_request)
{
  m_stat->enable(enable_debug_request);
  //if (!m_enable_debug_request)
  //info() << "WARNING: Mpi adpater debug request is disabled (multi-threading)";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
_checkFatalInRequest()
{
  if (isRequestErrorAreFatal())
    ARCCORE_FATAL("Error in requests management");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
setMpiProfiling(IMpiProfiling* mpi_profiling)
{
  m_mpi_prof = mpi_profiling;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMpiProfiling* MpiAdapter::
getMpiProfiling() const
{
  return m_mpi_prof;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
setProfiler(IProfiler* profiler)
{
  if (!profiler) {
    m_mpi_prof = nullptr;
    return;
  }

  IMpiProfiling* p = dynamic_cast<IMpiProfiling*>(profiler);
  if (!p)
    ARCCORE_FATAL("Invalid profiler. Profiler has to implemented interface 'IMpiProfiling'");
  m_mpi_prof = p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IProfiler* MpiAdapter::
profiler() const
{
  return m_mpi_prof;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention : Non thread-safe !
void MpiAdapter::
initializeWindowCreator(MPI_Comm comm_machine)
{
  if (m_window_creator.isNull()) {
    Integer machine_comm_rank = 0;
    Integer machine_comm_size = 0;
    ::MPI_Comm_rank(comm_machine, &machine_comm_rank);
    ::MPI_Comm_size(comm_machine, &machine_comm_size);
    m_window_creator = makeRef(new MpiMachineShMemWinBaseInternalCreator(comm_machine, machine_comm_rank, machine_comm_size, m_communicator, m_comm_size));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMachineShMemWinBaseInternalCreator* MpiAdapter::
windowCreator() const
{
  return m_window_creator.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
