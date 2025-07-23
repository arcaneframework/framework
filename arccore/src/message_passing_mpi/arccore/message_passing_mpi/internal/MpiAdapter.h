// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiAdapter.h                                                (C) 2000-2025 */
/*                                                                           */
/* Implémentation des messages avec MPI.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPIADAPTER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPIADAPTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceAccessor.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing_mpi/internal/MessagePassingMpiEnum.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/message_passing/Request.h"
#include "arccore/collections/CollectionsGlobal.h"

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{
class MpiMachineMemoryWindowBaseInternalCreator;

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
, public IRequestCreator
{
 public:
  class RequestSet;
  struct SubRequestInfo;
 public:

  MpiAdapter(ITraceMng* msg,IStat* stat,
             MPI_Comm comm,MpiLock* mpi_lock,
             IMpiProfiling* mpi_prof = nullptr);
  MpiAdapter(const MpiAdapter& rhs) = delete;
  MpiAdapter& operator=(const MpiAdapter& rhs) = delete;

 protected:

  ~MpiAdapter() override;

 public:

  //! Détruit l'instance. Elle ne doit plus être utilisée par la suite.
  void destroy();

 public:

  void broadcast(void* buf,Int64 nb_elem,Int32 root,MPI_Datatype datatype);
  void allGather(const void* send_buf,void* recv_buf,
                 Int64 nb_elem,MPI_Datatype datatype);
  void gather(const void* send_buf,void* recv_buf,
              Int64 nb_elem,Int32 root,MPI_Datatype datatype);
  void allGatherVariable(const void* send_buf,void* recv_buf,const int* recv_counts,
                         const int* recv_indexes,Int64 nb_elem,MPI_Datatype datatype);
  void gatherVariable(const void* send_buf,void* recv_buf,const int* recv_counts,
                      const int* recv_indexes,Int64 nb_elem,Int32 root,MPI_Datatype datatype);
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
  
  //! Version non bloquante de send sans statistique temporelle
  Request sendNonBlockingNoStat(const void* send_buffer,Int64 send_buffer_size,
                                Int32 proc,MPI_Datatype data_type,int mpi_tag);

  Request directRecv(void* recv_buffer,Int64 recv_buffer_size,
                     Int32 source_rank,Int64 elem_size,MPI_Datatype data_type,
                     int mpi_tag,bool is_blocked);

  //! Version non bloquante de receive sans statistiques temporelles
  Request receiveNonBlockingNoStat(void* recv_buffer,Int64 recv_buffer_size,
                                   Int32 source_rank,MPI_Datatype data_type,int mpi_tag);

  Request directSendPack(const void* send_buffer,Int64 send_buffer_size,
                         Int32 proc,int mpi_tag,bool is_blocked);

  void  probeRecvPack(UniqueArray<Byte>& recv_buffer,Int32 proc);

  MessageId probeMessage(PointToPointMessageInfo message);

  MessageSourceInfo legacyProbeMessage(PointToPointMessageInfo message);

  Request directRecv(void* recv_buffer,Int64 recv_buffer_size,
                     MessageId message,Int64 elem_size,MPI_Datatype data_type,
                     bool is_blocked);

  Request directRecvPack(void* recv_buffer,Int64 recv_buffer_size,
                         Int32 proc,int mpi_tag,bool is_blocking);

  void waitAllRequests(ArrayView<Request> requests);


 private:
  bool _waitAllRequestsMPI(ArrayView<Request> requests,ArrayView<bool> indexes,
                           ArrayView<MPI_Status> mpi_status);
 public:
  void waitSomeRequests(ArrayView<Request> requests,
                        ArrayView<bool> indexes,
                        bool is_non_blocking);

  void waitSomeRequestsMPI(ArrayView<Request> requests,
                           ArrayView<bool> indexes,
                           ArrayView<MPI_Status> mpi_status,bool is_non_blocking);
 public:
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

  //! Construit une requête Arccore à partir d'une requête MPI.
  Request buildRequest(int ret,MPI_Request request);

 public:

  //! Indique si les erreurs dans la liste des requêtes sont fatales
  void setRequestErrorAreFatal(bool v);
  bool isRequestErrorAreFatal() const;

  //! Indique si on affiche des messages pour les erreurs dans les requêtes.
  void setPrintRequestError(bool v);
  bool isPrintRequestError() const;

  //! Indique si on affiche des messages pour chaque appel MPI.
  void setTraceMPIMessage(bool v) { m_is_trace = v; }
  bool isTraceMPIMessage() const { return m_is_trace; }

  /*!
   * \brief Indique si on vérifie les requêtes.
   *
   * Cette valeur ne doit être modifiée s'il y a des requêtes en cours.
   */
  void setCheckRequest(bool v);
  bool isCheckRequest() const;

 public:

  void setMpiProfiling(IMpiProfiling* mpi_profiling);
  void setProfiler(IProfiler* profiler);
	IMpiProfiling* getMpiProfiling() const;
	IProfiler* profiler() const;

 public:

  ITimeMetricCollector* timeMetricCollector() const { return m_metric_collector; }
  void setTimeMetricCollector(ITimeMetricCollector* v) { m_metric_collector = v; }

  bool isAllowNullRankForAnySource() const { return m_is_allow_null_rank_for_any_source; }

 public:

  // void* createAllInOneMachineMemoryWindowBase(Integer sizeof_local) const;
  // void freeAllInOneMachineMemoryWindowBase(void* aio_node_window) const;
  // Ref<MpiAllInOneMachineMemoryWindowBase> readAllInOneMachineMemoryWindowBase(void* aio_node_window) const
  // {
  //   return makeRef(new MpiAllInOneMachineMemoryWindowBase(aio_node_window, sizeof(MPI_Win), m_machine_communicator, m_machine_comm_rank));
  // }

  MpiMachineMemoryWindowBaseInternalCreator* windowCreator();

 private:

  IStat* m_stat = nullptr;
  MpiLock* m_mpi_lock = nullptr;
  IMpiProfiling* m_mpi_prof = nullptr;
  ITimeMetricCollector* m_metric_collector = nullptr;
  MPI_Comm m_communicator; //!< Communicateur MPI
  int m_comm_rank = A_PROC_NULL_RANK;
  int m_comm_size = 0;
  MPI_Comm m_machine_communicator; //!< Communicateur MPI sur le noeud de calcul
  int m_machine_comm_rank = A_PROC_NULL_RANK;
  int m_machine_comm_size = 0;
  Int64 m_nb_all_reduce = 0;
  Int64 m_nb_reduce = 0;
  bool m_is_trace = false;
  RequestSet* m_request_set = nullptr;
  //! Requêtes vides. Voir MpiAdapter.cc pour plus d'infos.
  MPI_Request m_empty_request1;
  MPI_Request m_empty_request2;
  int m_recv_buffer_for_empty_request[1];
  int m_send_buffer_for_empty_request2[1];
  int m_recv_buffer_for_empty_request2[1];

  // Si vrai, autorise d'utiliser le rang nul (A_NULL_RANK) pour spécifier MPI_ANY_SOURCE
  // Cela est le défaut dans les versions de Arccore avant juillet 2024.
  // A partir de 2025 il faudra l'interdire.
  // La variable d'environnement ARCCORE_ALLOW_NULL_RANK_FOR_MPI_ANY_SOURCE permettra
  // de temporairement garder un mode compatible.
  bool m_is_allow_null_rank_for_any_source = true;

  MpiMachineMemoryWindowBaseInternalCreator* m_window_creator;

 private:
  
  void _trace(const char* function);
  void _addRequest(MPI_Request request);
  void _removeRequest(MPI_Request request);
  void _checkFatalInRequest();
  MessageId _probeMessage(MessageRank source,MessageTag tag,bool is_blocking);
  MessageSourceInfo _legacyProbeMessage(MessageRank source,MessageTag tag,bool is_blocking);
  bool _handleEndRequests(ArrayView<Request> requests,ArrayView<bool> done_indexes,
                          ArrayView<MPI_Status> status);
  void _checkHasNoRequests();
  MessageSourceInfo _buildSourceInfoFromStatus(const MPI_Status& status);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
