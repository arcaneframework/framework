// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiAdapter.h                                                (C) 2000-2019 */
/*                                                                           */
/* Implémentation des messages avec MPI.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPIADAPTER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPIADAPTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceAccessor.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing_mpi/MessagePassingMpiEnum.h"
#include "arccore/message_passing_mpi/IMpiProfiling.h"
#include "arccore/collections/CollectionsGlobal.h"

#include "arccore/base/BaseTypes.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
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

  MpiAdapter(ITraceMng* msg,IStat* stat,MPI_Comm comm,MpiLock* mpi_lock,
             IMpiProfiling* mpi_prof = nullptr);

 protected:

  ~MpiAdapter();

 public:

  //! Détruit l'instance. Elle ne doit plus être utilisée par la suite.
  void destroy();

 public:

  void broadcast(void* buf,Int64 nb_elem,Int32 root,MPI_Datatype datatype);
  void allGather(const void* send_buf,void* recv_buf,
                 Int64 nb_elem,MPI_Datatype datatype);
  void gather(const void* send_buf,void* recv_buf,
              Int64 nb_elem,Int32 root,MPI_Datatype datatype);
  void allGatherVariable(const void* send_buf,void* recv_buf,int* recv_counts,
                         int* recv_indexes,Int64 nb_elem,MPI_Datatype datatype);
  void gatherVariable(const void* send_buf,void* recv_buf,int* recv_counts,
                      int* recv_indexes,Int64 nb_elem,Int32 root,MPI_Datatype datatype);
  void scatterVariable(const void* send_buf,const int* send_count,const int* send_indexes,
                       void* recv_buf,Int64 nb_elem,Int32 root,MPI_Datatype datatype);
  void allToAll(const void* send_buf,void* recv_buf,Int32 count,MPI_Datatype datatype);
  void allToAllVariable(const void* send_buf,const int* send_counts,
                        const int* send_indexes,void* recv_buf,const int* recv_counts,
                        const int* recv_indexes,MPI_Datatype datatype);
  void reduce(const void* send_buf,void* recv_buf,Int64 count,MPI_Datatype datatype,MPI_Op op,Int32 root);
  void allReduce(const void* send_buf,void* recv_buf,Int64 count,MPI_Datatype datatype,MPI_Op op);
  void scan(const void* send_buf,void* recv_buf,Int64 count,MPI_Datatype datatype,MPI_Op op);
  void directSendRecv(const void* send_buffer,Int64 send_buffer_size,
                      void* recv_buffer,Int64 recv_buffer_size,
                      Int32 proc,Int64 elem_size,MPI_Datatype data_type);

  Request directSend(const void* send_buffer,Int64 send_buffer_size,
                     Int32 proc,Int64 elem_size,MPI_Datatype data_type,
                     int mpi_tag,bool is_blocked);
  
  Request directRecv(void* recv_buffer,Int64 recv_buffer_size,
                     Int32 proc,Int64 elem_size,MPI_Datatype data_type,
                     int mpi_tag,bool is_blocked);

  Request directSendPack(const void* send_buffer,Int64 send_buffer_size,
                         Int32 proc,int mpi_tag,bool is_blocked);

  void  probeRecvPack(UniqueArray<Byte>& recv_buffer,Int32 proc);

  Request directRecvPack(void* recv_buffer,Int64 recv_buffer_size,
                         Int32 proc,int mpi_tag,bool is_blocking);

  void waitAllRequests(ArrayView<Request> requests);

  void waitAllRequestsMPI(ArrayView<Request> requests,
                       ArrayView<bool> indexes,
                       ArrayView<MPI_Status> mpi_status);

  void waitSomeRequests(ArrayView<Request> requests,
                        ArrayView<bool> indexes,
                        bool is_non_blocking);

  void waitSomeRequestsMPI(ArrayView<Request> requests,
                        ArrayView<bool> indexes,
                        ArrayView<MPI_Status> mpi_status,bool is_non_blocking);

  //! Rang de cette instance dans le communicateur
  int commRank() const { return m_comm_rank; }

  //! Nombre de rangs dans le communicateur
  int commSize() const { return m_comm_size; }

  MpiMessagePassingMng* commSplit(bool keep);

  void freeRequest(Request& request);
  bool testRequest(Request& request);

  void enableDebugRequest(bool enable_debug_request);

  MpiLock* mpiLock() const { return m_mpi_lock; }

  Request nonBlockingBroadcast(void* buf,Int64 nb_elem,Int32 root,MPI_Datatype datatype);
  Request nonBlockingAllGather(const void* send_buf,void* recv_buf,Int64 nb_elem,MPI_Datatype datatype);
  Request nonBlockingGather(const void* send_buf,void* recv_buf,Int64 nb_elem,Int32 root,MPI_Datatype datatype);

  Request nonBlockingAllToAll(const void* send_buf,void* recv_buf,Int32 count,MPI_Datatype datatype);
  Request nonBlockingAllReduce(const void* send_buf,void* recv_buf,Int64 count,MPI_Datatype datatype,MPI_Op op);
  Request nonBlockingAllToAllVariable(const void* send_buf,const int* send_counts,
                                      const int* send_indexes,void* recv_buf,const int* recv_counts,
                                      const int* recv_indexes,MPI_Datatype datatype);

  Request nonBlockingBarrier();
  void barrier();

  int toMPISize(Int64 count);

 public:

  //! Indique si les erreurs dans la liste des requêtes sont fatales
  void setRequestErrorAreFatal(bool v) { m_request_error_is_fatal = v; }
  bool isRequestErrorAreFatal() const { return m_request_error_is_fatal; }

  //! Indique si on affiche des messages pour les erreurs dans les requêtes.
  void setPrintRequestError(bool v) { m_is_report_error_in_request = v; }
  bool isPrintRequestError() const { return m_is_report_error_in_request; }

  //! Indique si on affiche des messages pour chaque appel MPI.
  void setTraceMPIMessage(bool v) { m_is_trace = v; }
  bool isTraceMPIMessage() const { return m_is_trace; }

 public:

  void setMpiProfiling(IMpiProfiling* mpi_profiling);
	IMpiProfiling* getMpiProfiling();

 private:

  IStat* m_stat;
  MpiLock* m_mpi_lock;
  IMpiProfiling* m_mpi_prof;
  MPI_Comm m_communicator; //!< Communicateur MPI
  int m_comm_rank;
  int m_comm_size;
  Int64 m_nb_all_reduce = 0;
  Int64 m_nb_reduce = 0;
  bool m_is_trace = false;
  std::set<MPI_Request> m_allocated_requests;
  bool m_request_error_is_fatal;
  bool m_is_report_error_in_request;
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

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
