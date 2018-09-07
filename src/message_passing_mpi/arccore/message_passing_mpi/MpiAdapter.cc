/*---------------------------------------------------------------------------*/
/* MpiAdapter.cc                                               (C) 2000-2018 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiLock.h"

#include "arccore/trace/ITraceMng.h"

#include "arccore/collections/Array.h"

#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/IStat.h"

#include "arccore/base/IStackTraceService.h"
#include "arccore/base/TimeoutException.h"
#include "arccore/base/String.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/TraceInfo.h"

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

MpiAdapter::
MpiAdapter(ITraceMng* trace,IStat* stat,MPI_Comm comm,MpiLock* mpi_lock)
: TraceAccessor(trace)
, m_stat(stat)
, m_mpi_lock(mpi_lock)
, m_communicator(comm)
, m_comm_rank(0)
, m_comm_size(0)
, m_nb_all_reduce(0)
, m_nb_reduce(0)
, m_is_trace(false)
, m_request_error_is_fatal(false)
, m_empty_request(MPI_REQUEST_NULL)
{
  if (Platform::getEnvironmentVariable("ARCCORE_TRACE_MPI")=="TRUE")
    m_is_trace = true;

  ::MPI_Comm_rank(m_communicator,&m_comm_rank);
  ::MPI_Comm_size(m_communicator,&m_comm_size);

  /*!
   * Ce type de requête est utilisé par openmpi à partir de
   * la version 1.8 (il faut voir pour la 1.6, sachant que la 1.4 et 1.5
   * ne l'ont pas). Cette requête fonctionne un peu comme MPI_REQUEST_NULL
   * et il est possible qu'elle soit retournée plusieurs fois. Il ne faut
   * donc pas mettre cette requête dans m_allocated_requests.
   * On ne peut pas accéder directement à l'adresse de cette requête vide
   * mais l'implémentation 1.8 de openmpi retourne cette requête lorsqu'on
   * appelle un IRecv avec une source MPI_PROC_NULL. On récupère donc la valeur
   * comme cela.
   */
  MPI_Irecv(m_recv_buffer_for_empty_request,1,MPI_INT,MPI_PROC_NULL,50505,m_communicator,&m_empty_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiAdapter::
~MpiAdapter()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
destroy()
{
  // On ne peut pas faire ce test dans le destructeur car il peut
  // potentiellement lancé une exception et cela ne doit pas être fait
  // dans un destructeur.
  size_t nb_request = m_allocated_requests.size();
  if (nb_request!=0){
    warning() << " Pending mpi requests size=" << nb_request;
    for( std::set<MPI_Request>::const_iterator i = m_allocated_requests.begin();
         i!=m_allocated_requests.end(); ++i ){
    }
    _checkFatalInRequest();
  }
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
_trace(const char* function)
{
  if (m_is_trace) {
    IStackTraceService* stack_service = Platform::getStackTraceService();
    if (stack_service)
      info() << "MPI_TRACE: " << function << "\n" << stack_service->stackTrace().toString();
    else
      info() << "MPI_TRACE: " << function;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
broadcast(void* buf,Integer nb_elem,Integer root,MPI_Datatype datatype)
{
  int _nb_elem = static_cast<int>(nb_elem);
  _trace(" MPI_Bcast");
  double begin_time = MPI_Wtime();
  int r = MPI_Bcast(buf,_nb_elem,datatype,root,m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("Broadcast",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingBroadcast(void* buf,Integer nb_elem,Integer root,MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
#ifdef ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE
  int _nb_elem = static_cast<int>(nb_elem);
  _trace(" MPI_Bcast");
  double begin_time = MPI_Wtime();
  ret = MPI_Ibcast(buf,_nb_elem,datatype,root,m_communicator,&mpi_request);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("IBroadcast",sr_time,0);
  _addRequest(mpi_request);
#else
  ret = broadcast(buf,nb_elem,root,datatype);
#endif
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
gather(const void* send_buf,void* recv_buf,
       Integer nb_elem,Integer root,MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = static_cast<int>(nb_elem);
  int _root = static_cast<int>(root);
  _trace("MPI_Gather");
  double begin_time = MPI_Wtime();
  int r = MPI_Gather(_sbuf,_nb_elem,datatype,
                     recv_buf,_nb_elem,datatype,_root,
                     m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("Gather",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingGather(const void* send_buf,void* recv_buf,
                  Integer nb_elem,Integer root,MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
#ifdef ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = static_cast<int>(nb_elem);
  int _root = static_cast<int>(root);
  _trace("MPI_Igather");
  double begin_time = MPI_Wtime();
  ret = MPI_Igather(_sbuf,_nb_elem,datatype,recv_buf,_nb_elem,datatype,_root,
                    m_communicator,&mpi_request);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("IGather",sr_time,0);
  _addRequest(mpi_request);
#else
  ret = gather(send_buf,recv_buf,nb_elem,root,datatype);
#endif
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
allGather(const void* send_buf,void* recv_buf,
           Integer nb_elem,MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = static_cast<int>(nb_elem);
  _trace("MPI_Allgather");
  double begin_time = MPI_Wtime();
  int r = MPI_Allgather(_sbuf,_nb_elem,datatype,recv_buf,_nb_elem,datatype,
                        m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("AllGather",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllGather(const void* send_buf,void* recv_buf,
                     Integer nb_elem,MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
#ifdef ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = static_cast<int>(nb_elem);
  _trace("MPI_Iallgather");
  double begin_time = MPI_Wtime();
  ret = MPI_Iallgather(_sbuf,_nb_elem,datatype,recv_buf,_nb_elem,datatype,
                       m_communicator,&mpi_request);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("IAllGather",sr_time,0);
  _addRequest(mpi_request);
#else
  ret = allGather(send_buf,recv_buf,nb_elem,datatype);
#endif
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
gatherVariable(const void* send_buf,void* recv_buf,int* recv_counts,
               int* recv_indexes,Integer nb_elem,Integer root,MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _nb_elem = static_cast<int>(nb_elem);
  int _root = static_cast<int>(root);
  _trace(" MPI_Gatherv");
  double begin_time = MPI_Wtime();
  int r = MPI_Gatherv(_sbuf,_nb_elem,datatype,
                      recv_buf,recv_counts,recv_indexes,datatype,
                      _root,m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("Gather",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
allGatherVariable(const void* send_buf,void* recv_buf,int* recv_counts,
                  int* recv_indexes,Integer nb_elem,MPI_Datatype datatype)
{
  void* _sbuf    = const_cast<void*>(send_buf);
  int   _nb_elem = static_cast<int>(nb_elem);
  _trace("MPI_Allgatherv");
  //info() << " ALLGATHERV N=" << _nb_elem;
  //for( int i=0; i<m_comm_size; ++i )
  //info() << " ALLGATHERV I=" << i << " recv_count=" << recv_counts[i]
  //     << " recv_indexes=" << recv_indexes[i];
  double begin_time = MPI_Wtime();
  int r = MPI_Allgatherv(_sbuf,_nb_elem,datatype,
                        recv_buf,recv_counts,recv_indexes,datatype,
                        m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("Gather",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
scatterVariable(const void* send_buf,const int* send_count,const int* send_indexes,
                void* recv_buf,Integer nb_elem,Integer root,MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int* _send_count = const_cast<int*>(send_count);
  int* _send_indexes = const_cast<int*>(send_indexes);
  int _nb_elem = static_cast<int>(nb_elem);

  _trace("MPI_Scatterv");
  double begin_time = MPI_Wtime();
  int r = MPI_Scatterv(_sbuf,_send_count,_send_indexes,datatype,recv_buf,
                       _nb_elem,datatype,root,m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("Scatter",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
allToAll(const void* send_buf,void* recv_buf,Integer count,MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int icount = static_cast<int>(count);
  _trace("MPI_Alltoall");
  double begin_time = MPI_Wtime();
  int r = MPI_Alltoall(_sbuf,icount,datatype,recv_buf,icount,datatype,m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("AllToAll",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllToAll(const void* send_buf,void* recv_buf,Integer count,MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
#ifdef ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE
  void* _sbuf = const_cast<void*>(send_buf);
  int icount = static_cast<int>(count);
  _trace("MPI_IAlltoall");
  double begin_time = MPI_Wtime();
  ret = MPI_Ialltoall(_sbuf,icount,datatype,recv_buf,icount,datatype,m_communicator,&mpi_request);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("IAllToAll",sr_time,0);
  _addRequest(mpi_request);
#else
  ret = allToAll(send_buf,recv_buf,count,datatype);
#endif
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
allToAllVariable(const void* send_buf,const int* send_counts,
                 const int* send_indexes,void* recv_buf,const int* recv_counts,
                 const int* recv_indexes,MPI_Datatype datatype)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int* _send_counts = const_cast<int*>(send_counts);
  int* _send_indexes = const_cast<int*>(send_indexes);
  int* _recv_counts = const_cast<int*>(recv_counts);
  int* _recv_indexes = const_cast<int*>(recv_indexes);

  _trace("MPI_Alltoallv");
  double begin_time = MPI_Wtime();
  int r = MPI_Alltoallv(_sbuf,_send_counts,_send_indexes,datatype,
                        recv_buf,_recv_counts,_recv_indexes,datatype,
                        m_communicator);
  double end_time = MPI_Wtime();
  double sr_time   = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("AllToAll",sr_time,0);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllToAllVariable(const void* send_buf,const int* send_counts,
                            const int* send_indexes,void* recv_buf,const int* recv_counts,
                            const int* recv_indexes,MPI_Datatype datatype)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
#ifdef ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE
  void* _sbuf = const_cast<void*>(send_buf);
  int* _send_counts = const_cast<int*>(send_counts);
  int* _send_indexes = const_cast<int*>(send_indexes);
  int* _recv_counts = const_cast<int*>(recv_counts);
  int* _recv_indexes = const_cast<int*>(recv_indexes);

  _trace("MPI_Ialltoallv");
  double begin_time = MPI_Wtime();
  ret = MPI_Ialltoallv(_sbuf,_send_counts,_send_indexes,datatype,
                       recv_buf,_recv_counts,_recv_indexes,datatype,
                       m_communicator,&mpi_request);
  double end_time = MPI_Wtime();
  double sr_time = (end_time-begin_time);
  //TODO determiner la taille des messages
  m_stat->add("IAllToAll",sr_time,0);
  _addRequest(mpi_request);
#else
  ret = allToAllVariable(send_buf,send_counts,send_indexes,recv_buf,recv_counts,
                         recv_indexes,datatype);
#endif
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingBarrier()
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
#ifdef ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE
  ret = MPI_Ibarrier(m_communicator,&mpi_request);
  _addRequest(mpi_request);
#else
  MPI_Barrier(m_communicator);
#endif
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
allReduce(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = static_cast<int>(n);
  double begin_time = MPI_Wtime();
  _trace("MPI_Allreduce");
  int ret = 0;
  try{
    ++m_nb_all_reduce;
    ret = MPI_Allreduce(_sbuf,recv_buf,_n,datatype,op,m_communicator);
  }
  catch(TimeoutException& ex)
  {
    std::ostringstream ostr;
    ostr << "MPI_Allreduce"
         << " send_buf=" << send_buf
         << " recv_buf=" << recv_buf
         << " n=" << n
         << " datatype=" << datatype
         << " op=" << op
         << " NB=" << m_nb_all_reduce;
    ex.setAdditionalInfo(ostr.str());
    throw;
  }
  double end_time = MPI_Wtime();
  m_stat->add("Reduce",end_time-begin_time,n);
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
nonBlockingAllReduce(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op)
{
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = -1;
#ifdef ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = static_cast<int>(n);
  double begin_time = MPI_Wtime();
  _trace("MPI_IAllreduce");
  ret = MPI_Iallreduce(_sbuf,recv_buf,_n,datatype,op,m_communicator,&mpi_request);
  double end_time = MPI_Wtime();
  m_stat->add("IReduce",end_time-begin_time,n);
  _addRequest(mpi_request);
#else
  ret = allReduce(send_buf,recv_buf,n,datatype,op);
#endif
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
reduce(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op,Integer root)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = static_cast<int>(n);
  int _root = static_cast<int>(root);
  double begin_time = MPI_Wtime();
  _trace("MPI_reduce");
  int ret = 0;
  try{
    ++m_nb_reduce;
    ret = MPI_Reduce(_sbuf,recv_buf,_n,datatype,op,_root,m_communicator);
  }
  catch(TimeoutException& ex)
  {
    std::ostringstream ostr;
    ostr << "MPI_reduce"
         << " send_buf=" << send_buf
         << " recv_buf=" << recv_buf
         << " n=" << n
         << " datatype=" << datatype
         << " op=" << op
         << " root=" << root
         << " NB=" << m_nb_reduce;
    ex.setAdditionalInfo(ostr.str());
    throw;
  }
  
  double end_time = MPI_Wtime();
  m_stat->add("Reduce",end_time-begin_time,0);
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
scan(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op)
{
  void* _sbuf = const_cast<void*>(send_buf);
  int _n = static_cast<int>(n);
  double begin_time = MPI_Wtime();
  _trace("MPI_Scan");
  int ret = MPI_Scan(_sbuf,recv_buf,_n,datatype,op,m_communicator);
  double end_time = MPI_Wtime();
  m_stat->add("Scan",end_time-begin_time,0);
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MpiAdapter::
directSendRecv(const void* send_buffer,Integer send_buffer_size,
               void* recv_buffer,Integer recv_buffer_size,
               Integer proc,Integer elem_size,MPI_Datatype data_type
               )
{
  void* v_send_buffer = const_cast<void*>(send_buffer);
  MPI_Status mpi_status;
  double begin_time = MPI_Wtime();
  _trace("MPI_Sendrecv");
  int ret = MPI_Sendrecv(v_send_buffer,send_buffer_size,
                         data_type,proc,99,
                         recv_buffer,recv_buffer_size,
                         data_type,proc,99,
                         m_communicator,&mpi_status);
  double end_time = MPI_Wtime();
  size_t send_size = send_buffer_size * elem_size;
  size_t recv_size = recv_buffer_size * elem_size;
  double sr_time   = (end_time-begin_time);

  //debug(Trace::High) << "MPI SendRecv: send " << send_size << " recv "
  //                      << recv_size << " time " << sr_time ;
  m_stat->add("SendRecv",sr_time,send_size+recv_size);
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directSend(const void* send_buffer,Integer send_buffer_size,
           Integer proc,Integer elem_size,MPI_Datatype data_type,
           int mpi_tag,bool is_blocked
           )
{
  void* v_send_buffer = const_cast<void*>(send_buffer);
  MPI_Request mpi_request = MPI_REQUEST_NULL;

  double begin_time = 0.0;
  double end_time = 0.0;
  size_t send_size = send_buffer_size * elem_size;
  int ret = 0;
  if (m_is_trace)
    info() << "MPI_TRACE: MPI Send: send before"
           << " size=" << send_size
           << " dest=" << proc
           << " tag=" << mpi_tag
           << " datatype=" << data_type
           << " blocking " << is_blocked;
  if (is_blocked){
    // si m_mpi_lock n'est pas nul, il faut
    // utiliser un MPI_ISend suivi d'une boucle
    // active de MPI_Test pour eviter tout probleme
    // de dead lock.
    if (m_mpi_lock){
      {
        MpiLock::Section mls(m_mpi_lock);
        begin_time = MPI_Wtime();
        ret = MPI_Isend(v_send_buffer,send_buffer_size,
                        data_type,proc,mpi_tag,m_communicator,&mpi_request);
      }
      int is_finished = 0;
      MPI_Status mpi_status;
      while (is_finished==0){
        MpiLock::Section mls(m_mpi_lock);
        MPI_Request_get_status(mpi_request,&is_finished,&mpi_status); 
        if (is_finished!=0){
          MPI_Wait(&mpi_request,(MPI_Status*)MPI_STATUS_IGNORE);
          end_time = MPI_Wtime();
          mpi_request = MPI_REQUEST_NULL;
        }
      }
    }
    else{
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      ret = MPI_Send(v_send_buffer,send_buffer_size,
                     data_type,proc,mpi_tag,m_communicator);
      end_time = MPI_Wtime();
    }
  }
  else{
    {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      ret = MPI_Isend(v_send_buffer,send_buffer_size,
                      data_type,proc,mpi_tag,m_communicator,&mpi_request);
      if (m_is_trace)
        info() << " ISend ret=" << ret << " proc=" << proc << " request=" << mpi_request;
      end_time = MPI_Wtime();
      _addRequest(mpi_request);
    }
    if (m_is_trace){
      info() << "MPI Send: send after"
             << " request=" << mpi_request;
    }
  }
  double sr_time   = (end_time-begin_time);
  
  debug(Trace::High) << "MPI Send: send " << send_size
                     << " time " << sr_time << " blocking " << is_blocked;
  m_stat->add("Send",end_time-begin_time,send_size);
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directSendPack(const void* send_buffer,Integer send_buffer_size,
               Integer proc,int mpi_tag,bool is_blocked
            )
{
  return directSend(send_buffer,send_buffer_size,proc,1,MPI_PACKED,mpi_tag,is_blocked);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directRecv(void* recv_buffer,Integer recv_buffer_size,
           Integer proc,Integer elem_size,MPI_Datatype data_type,
           int mpi_tag,bool is_blocked
           )
{
  MPI_Status  mpi_status;
  MPI_Request mpi_request = MPI_REQUEST_NULL;
  int ret = 0;
  double begin_time = 0.0;
  double end_time = 0.0;
  
  int i_proc = 0;
  if (proc==A_NULL_RANK)
    i_proc = MPI_ANY_SOURCE;
  else
    i_proc = static_cast<int>(proc);

  size_t recv_size = recv_buffer_size * elem_size;
  if (m_is_trace){
    info() << "MPI_TRACE: MPI Recv: recv before "
           << " size=" << recv_size
           << " from=" << i_proc
           << " tag=" << mpi_tag
           << " datatype=" << data_type
           << " blocking=" << is_blocked;
  }
  if (is_blocked){
    // si m_mpi_lock n'est pas nul, il faut
    // utiliser un MPI_IRecv suivi d'une boucle
    // active de MPI_Test pour eviter tout probleme
    // de dead lock.
    if (m_mpi_lock){
      {
        MpiLock::Section mls(m_mpi_lock);
        begin_time = MPI_Wtime();
        ret = MPI_Irecv(recv_buffer,recv_buffer_size,
                        data_type,i_proc,mpi_tag,
                        m_communicator,&mpi_request);
      }
      int is_finished = 0;
      MPI_Status mpi_status;
      while (is_finished==0){
        MpiLock::Section mls(m_mpi_lock);
        MPI_Request_get_status(mpi_request,&is_finished,&mpi_status); 
        if (is_finished!=0){
          end_time = MPI_Wtime();
          MPI_Wait(&mpi_request,(MPI_Status*)MPI_STATUS_IGNORE);
          mpi_request = MPI_REQUEST_NULL;
        }
      }
    }
    else{
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      ret = MPI_Recv(recv_buffer,recv_buffer_size,
                     data_type,i_proc,mpi_tag,
                     m_communicator,&mpi_status);
      end_time = MPI_Wtime();
    }
  }
  else{
    {
      MpiLock::Section mls(m_mpi_lock);
      begin_time = MPI_Wtime();
      ret = MPI_Irecv(recv_buffer,recv_buffer_size,
                      data_type,i_proc,mpi_tag,
                      m_communicator,&mpi_request);
      end_time = MPI_Wtime();
      _addRequest(mpi_request);
    }
    if (m_is_trace){
      info() << "MPI Recv: recv after "
             << " request=" << mpi_request;
    }
  }
  double sr_time   = (end_time-begin_time);
  
  debug(Trace::High) << "MPI Recv: recv after " << recv_size
                     << " time " << sr_time << " blocking " << is_blocked;
  m_stat->add("Recv",end_time-begin_time,recv_size);
  return Request(ret,mpi_request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
probeRecvPack(UniqueArray<Byte>& recv_buffer,Integer proc)
{
  double begin_time = MPI_Wtime();
  MPI_Status status;
  int recv_buffer_size = 0;
  _trace("MPI_Probe");
  MPI_Probe(proc,101,m_communicator,&status);
  MPI_Get_count(&status,MPI_PACKED,&recv_buffer_size);

  recv_buffer.resize(recv_buffer_size);
  MPI_Recv(recv_buffer.data(),recv_buffer_size,MPI_PACKED,proc,101,m_communicator,&status);

  double end_time = MPI_Wtime();
  size_t recv_size = recv_buffer_size;
  double sr_time   = (end_time-begin_time);
  debug(Trace::High) << "MPI probeRecvPack " << recv_size
                     << " time " << sr_time;
  m_stat->add("Recv",end_time-begin_time,recv_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiAdapter::
directRecvPack(void* recv_buffer,Integer recv_buffer_size,
               Integer proc,int mpi_tag,bool is_blocking)
{
  return directRecv(recv_buffer,recv_buffer_size,proc,1,MPI_PACKED,mpi_tag,is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
waitAllRequests(ArrayView<Request> requests,
                ArrayView<bool> indexes,
                ArrayView<MPI_Status> mpi_status)
{
  Integer size = requests.size();
  if (size==0)
    return;
  //ATTENTION: Mpi modifie en retour de MPI_Waitall ce tableau
  UniqueArray<MPI_Request> mpi_request(size);
  for( Integer i=0; i<size; ++i ){
    mpi_request[i] = (MPI_Request)(requests[i]);
    indexes[i] = true;
  }
  if (m_is_trace)
    info() << " MPI_waitall begin size=" << size;
  double diff_time = 0.0;
  if (m_mpi_lock){
    double begin_time = MPI_Wtime();
    for( Integer i=0; i<size; ++i ){
      MPI_Request request = (MPI_Request)(mpi_request[i]);
      int is_finished = 0;
      while (is_finished==0){
        MpiLock::Section mls(m_mpi_lock);
        MPI_Test(&request,&is_finished,(MPI_Status*)MPI_STATUS_IGNORE);
      }
    }
    double end_time = MPI_Wtime();
    diff_time = end_time - begin_time;
  }
  else{
    //TODO: transformer en boucle while et MPI_Testall si m_mpi_lock est non nul
    MpiLock::Section mls(m_mpi_lock);
    double begin_time = MPI_Wtime();
    MPI_Waitall(size,mpi_request.data(),mpi_status.data());
    double end_time = MPI_Wtime();
    diff_time = end_time - begin_time;
  }
  // Il ne faut pas utiliser mpi_request[i] car il est modifié par Mpi
  // mpi_request[i] == MPI_REQUEST_NULL
  for( Integer i=0; i<size; ++i ) {
    _removeRequest((MPI_Request)(requests[i]));
    requests[i].reset();
  }

  if (m_is_trace)
    info() << " MPI_waitall end size=" << size;
  m_stat->add("WaitAll",diff_time,size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
waitSomeRequests(ArrayView<Request> requests,ArrayView<bool> indexes,
                 ArrayView<MPI_Status> mpi_status,bool is_non_blocking)
{
  Integer size = requests.size();
  if (size==0)
    return;
  //TODO: utiliser des StackArray (quand ils seront disponibles...)
  UniqueArray<MPI_Request> mpi_request(size);
  UniqueArray<MPI_Request> saved_mpi_request(size);
  UniqueArray<int> completed_requests(size);
  int nb_completed_request = 0;
  for( Integer i=0; i<size; ++i ){
    // Sauve la requete pour la desallouer dans m_allocated_requests,
    // car sa valeur ne sera plus valide après appel à MPI_Wait*
    saved_mpi_request[i] = static_cast<MPI_Request>(requests[i]);
    //info() << " REQUEST_WAIT I=" << i << " M=" << size
    //<< " R=" << mpi_request[i];
  }

  double begin_time = MPI_Wtime();
  debug() << "WaitRequestBegin is_non_blocking=" << is_non_blocking;

  try{
    if (is_non_blocking){
      _trace("MPI_Testsome");
      {
        MpiLock::Section mls(m_mpi_lock);
        MPI_Testsome(size,saved_mpi_request.data(),&nb_completed_request,
                     completed_requests.data(),mpi_status.data());
      }
      //If there is no active handle in the list, it returns outcount = MPI_UNDEFINED.
      if (nb_completed_request == MPI_UNDEFINED) // Si aucune requete n'etait valide.
      	nb_completed_request = 0;
      debug() << "TestSome nb_completed=" << nb_completed_request;
    }
    else{
      _trace("MPI_Waitsome");
      {
        MpiLock::Section mls(m_mpi_lock);
        MPI_Waitsome(size,saved_mpi_request.data(),&nb_completed_request,
                     completed_requests.data(),mpi_status.data());
      }
      // Il ne faut pas utiliser mpi_request[i] car il est modifié par Mpi
      // mpi_request[i] == MPI_REQUEST_NULL
      if (nb_completed_request == MPI_UNDEFINED) // Si aucune requete n'etait valide.
      	nb_completed_request = 0;
      debug() << "WaitRequest nb_completed=" << nb_completed_request;
    }
  }
  catch(TimeoutException& ex)
  {
    std::ostringstream ostr;
    if (is_non_blocking)
      ostr << "MPI_Testsome";
    else
      ostr << "Mpi_Waitsome";
    ostr << " size=" << size
         << " is_non_blocking=" << is_non_blocking;
    ex.setAdditionalInfo(ostr.str());
    throw;
  }

  {
    MpiLock::Section mls(m_mpi_lock);
    for( int z=0; z<nb_completed_request; ++z ){
      int index = completed_requests[z];
      debug() << "Completed z=" << z
              << " index=" << index
              << " status=" << mpi_status[z].MPI_SOURCE
              << " status_index=" << mpi_status[index].MPI_SOURCE;
      indexes[index] = true;
      _removeRequest(static_cast<MPI_Request>(requests[index]));
      requests[index].reset();
    }
  }

  double end_time = MPI_Wtime();
  m_stat->add("WaitSome",end_time-begin_time,size);
  //return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiAdapter::
freeRequest(Request& request)
{
  if (!request.isValid()){
    warning() << "MpiAdapter::freeRequest() null request r=" << (MPI_Request)request;
    _checkFatalInRequest();
  }
  {
    MpiLock::Section mls(m_mpi_lock);

    MPI_Request mr = (MPI_Request)request;
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
  if (!request.isValid()){
    error() << "MpiAdapter::testRequest() null request r=" << (MPI_Request)request;
    return true;
  }

  MPI_Request mr = (MPI_Request)request;
  int is_finished = 0;

  {
    MpiLock::Section mls(m_mpi_lock);

    if (mr!=m_empty_request){
      // Il faut d'abord recuperer l'emplacement de la requete car si elle
      // est finie, elle sera automatiquement liberee par MPI lors du test.
      std::set<MPI_Request>::iterator ireq = m_allocated_requests.find(mr);
      if (ireq==m_allocated_requests.end()){
        error() << "MpiAdapter::testRequest() request not referenced "
                << " id=" << mr;
        _checkFatalInRequest();
      }
    }

    MPI_Test(&mr,&is_finished,(MPI_Status*)MPI_STATUS_IGNORE);
    //info() << "** TEST REQUEST r=" << mr << " is_finished=" << is_finished;
    if (is_finished!=0){
      _removeRequest(static_cast<MPI_Request>(request));
      request.reset();
      return true;
    }
  }

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \warning Cette fonction doit etre appelee avec le verrou mpi_lock actif.
 */
void MpiAdapter::
_addRequest(MPI_Request request)
{
  if (request==MPI_REQUEST_NULL){
    error() << "MpiAdapter::_addRequest() trying to add null request";
    _checkFatalInRequest();
    return;
  }
  if (request==m_empty_request)
    return;
  //info() << "MPI_ADAPTER:ADD REQUEST " << request;
  std::set<MPI_Request>::const_iterator i = m_allocated_requests.find(request);
  if (i!=m_allocated_requests.end()){
    error() << "MpiAdapter::_addRequest() request already referenced "
            << " id=" << request;
    _checkFatalInRequest();
    return;
  }
  m_allocated_requests.insert(request);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \warning Cette fonction doit etre appelee avec le verrou mpi_lock actif.
 */
void MpiAdapter::
_removeRequest(MPI_Request request)
{
  //info() << "MPI_ADAPTER:REMOVE REQUEST " << request;
  if (request==MPI_REQUEST_NULL){
    error() << "MpiAdapter::_removeRequest() null request (" << MPI_REQUEST_NULL << ")";
    _checkFatalInRequest();
    return;
  }
  if (request==m_empty_request)
    return;
  std::set<MPI_Request>::iterator i = m_allocated_requests.find(request);
  if (i==m_allocated_requests.end()){
    error() << "MpiAdapter::_removeRequest() request not referenced "
            << " id=" << request;
    _checkFatalInRequest();
  }
  else
    m_allocated_requests.erase(i);
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
  if (m_request_error_is_fatal)
    throw FatalErrorException(A_FUNCINFO,"Error in requests management");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
