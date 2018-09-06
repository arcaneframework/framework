/*---------------------------------------------------------------------------*/
/* MpiAdapter.h                                                (C) 2000-2018 */
/*                                                                           */
/* Implémentation des messages avec MPI.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPIADAPTER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPIADAPTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceAccessor.h"

#include "arccore/messagepassingmpi/MessagePassingMpiGlobal.h"

#include "arccore/collections/CollectionsGlobal.h"

#include "arccore/base/BaseTypes.h"

#include <set>

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
 * \brief Adapteur pour MPI.
 * \warning en hybride MPI/Thread, une instance de cette classe
 * est partagée entre tous les threads d'un processus MPI et donc
 * toutes les méthodes de cette classe doivent être thread-safe.
 * \todo rendre thread-safe les statistiques
 * \todo rendre thread-safe le m_allocated_request
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiAdapter
: public TraceAccessor
{
 public:
  
  //typedef Parallel::Request Request;
  //typedef Parallel::eReduceType eReduceType;

 public:

  MpiAdapter(ITraceMng* msg,IStat* stat,MPI_Comm comm,MpiLock* mpi_lock);

 protected:

  ~MpiAdapter();

 public:

  //! Détruit l'instance. Elle ne doit plus être utilisée par la suite.
  void destroy();

 public:

  int broadcast(void* buf,Integer nb_elem,Integer root,MPI_Datatype datatype);
  int allGather(const void* send_buf,void* recv_buf,
                Integer nb_elem,MPI_Datatype datatype);
  int gather(const void* send_buf,void* recv_buf,
             Integer nb_elem,Integer root,MPI_Datatype datatype);
  int allGatherVariable(const void* send_buf,void* recv_buf,int* recv_counts,
                        int* recv_indexes,Integer nb_elem,MPI_Datatype datatype);
  int gatherVariable(const void* send_buf,void* recv_buf,int* recv_counts,
                     int* recv_indexes,Integer nb_elem,Integer root,MPI_Datatype datatype);
  int scatterVariable(const void* send_buf,const int* send_count,const int* send_indexes,
                      void* recv_buf,Integer nb_elem,Integer root,MPI_Datatype datatype);
  int allToAll(const void* send_buf,void* recv_buf,Integer count,MPI_Datatype datatype);
  int allToAllVariable(const void* send_buf,const int* send_counts,
                       const int* send_indexes,void* recv_buf,const int* recv_counts,
                       const int* recv_indexes,MPI_Datatype datatype);
  int reduce(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op,Integer root);
  int allReduce(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op);
  int scan(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op);
  int directSendRecv(const void* send_buffer,Integer send_buffer_size,
                     void* recv_buffer,Integer recv_buffer_size,
                     Integer proc,Integer elem_size,MPI_Datatype data_type
                     );

  Request directSend(const void* send_buffer,Integer send_buffer_size,
                     Integer proc,Integer elem_size,MPI_Datatype data_type,
                     int mpi_tag,bool is_blocked
                     );
  
  Request directRecv(void* recv_buffer,Integer recv_buffer_size,
                     Integer proc,Integer elem_size,MPI_Datatype data_type,
                     int mpi_tag,bool is_blocked
                     );

  Request directSendPack(const void* send_buffer,Integer send_buffer_size,
                         Integer proc,int mpi_tag,bool is_blocked);

  void  probeRecvPack(UniqueArray<Byte>& recv_buffer,Integer proc);

  Request directRecvPack(void* recv_buffer,Integer recv_buffer_size,
                         Integer proc,int mpi_tag,bool is_blocking);

  void waitAllRequests(ArrayView<Request> requests,
                       ArrayView<bool> indexes,
                       ArrayView<MPI_Status> mpi_status);

  void waitSomeRequests(ArrayView<Request> requests,
                        ArrayView<bool> indexes,
                        ArrayView<MPI_Status> mpi_status,bool is_non_blocking);

  int commRank() const { return m_comm_rank; }
  int commSize() const { return m_comm_size; }

  void freeRequest(Request& request);
  bool testRequest(Request& request);

  void enableDebugRequest(bool enable_debug_request);

  MpiLock* mpiLock() const { return m_mpi_lock; }

  Request nonBlockingBroadcast(void* buf,Integer nb_elem,Integer root,MPI_Datatype datatype);
  Request nonBlockingAllGather(const void* send_buf,void* recv_buf,Integer nb_elem,MPI_Datatype datatype);
  Request nonBlockingGather(const void* send_buf,void* recv_buf,Integer nb_elem,Integer root,MPI_Datatype datatype);

  Request nonBlockingAllToAll(const void* send_buf,void* recv_buf,Integer count,MPI_Datatype datatype);
  Request nonBlockingAllReduce(const void* send_buf,void* recv_buf,Integer n,MPI_Datatype datatype,MPI_Op op);
  Request nonBlockingAllToAllVariable(const void* send_buf,const int* send_counts,
                                      const int* send_indexes,void* recv_buf,const int* recv_counts,
                                      const int* recv_indexes,MPI_Datatype datatype);

  Request nonBlockingBarrier();

 private:

  IStat* m_stat;
  MpiLock* m_mpi_lock;
  MPI_Comm m_communicator; //!< Communicateur MPI
  int m_comm_rank;
  int m_comm_size;
  Int64 m_nb_all_reduce;
  Int64 m_nb_reduce;
  bool m_is_trace;
  std::set<MPI_Request> m_allocated_requests;
  bool m_request_error_is_fatal;
  //! Requête vide. Voir MpiAdapter.cc pour plus d'infos.
  MPI_Request m_empty_request;
  int m_recv_buffer_for_empty_request[1];

 private:
  
  void _trace(const char* function);
  void _addRequest(MPI_Request request);
  void _removeRequest(MPI_Request request);
  void _checkFatalInRequest();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
