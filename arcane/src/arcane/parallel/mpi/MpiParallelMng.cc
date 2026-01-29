// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelMng.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/TimeoutException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Exception.h"
#include "arcane/utils/HPReal.h"

#include "arcane/core/IIOMng.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/parallel/IStat.h"
#include "arcane/core/internal/SerializeMessage.h"
#include "arcane/core/internal/ParallelMngInternal.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiParallelDispatch.h"
#include "arcane/parallel/mpi/MpiTimerMng.h"
#include "arcane/parallel/mpi/MpiSerializeMessage.h"
#include "arcane/parallel/mpi/MpiParallelNonBlockingCollective.h"
#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/mpi/IVariableSynchronizerMpiCommunicator.h"

#include "arcane/impl/ParallelReplication.h"
#include "arcane/impl/SequentialParallelMng.h"
#include "arcane/impl/internal/ParallelMngUtilsFactoryBase.h"
#include "arcane/impl/internal/VariableSynchronizer.h"

#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"
#include "arccore/message_passing_mpi/internal/MpiSerializeDispatcher.h"
#include "arccore/message_passing_mpi/internal/MpiRequestList.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arccore/message_passing_mpi/internal/MpiLock.h"
#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseInternalCreator.h"
#include "arccore/message_passing_mpi/internal/MpiMachineMemoryWindowBaseInternal.h"
#include "arccore/message_passing_mpi/internal/MpiDynamicMachineMemoryWindowBaseInternal.h"
#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/internal/SerializeMessageList.h"

#include "arcane_packages.h"

//#define ARCANE_TRACE_MPI

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arcane::MessagePassing;
using namespace Arcane::MessagePassing::Mpi;
using BasicSerializeMessage = Arcane::MessagePassing::internal::BasicSerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IIOMng*
arcaneCreateIOMng(IParallelMng* psm);

#if defined(ARCANE_HAS_MPI_NEIGHBOR)
// Défini dans MpiNeighborVariableSynchronizeDispatcher
extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiNeighborVariableSynchronizerFactory(MpiParallelMng* mpi_pm,
                                                   Ref<IVariableSynchronizerMpiCommunicator> synchronizer_communicator);
#endif
extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiBlockVariableSynchronizerFactory(MpiParallelMng* mpi_pm, Int32 block_size, Int32 nb_sequence);
extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiVariableSynchronizerFactory(MpiParallelMng* mpi_pm);
extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiDirectSendrecvVariableSynchronizerFactory(MpiParallelMng* mpi_pm);
extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiLegacyVariableSynchronizerFactory(MpiParallelMng* mpi_pm);
#if defined(ARCANE_HAS_PACKAGE_NCCL)
extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateNCCLVariableSynchronizerFactory(IParallelMng* mpi_pm);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiParallelMngBuildInfo::
MpiParallelMngBuildInfo(MPI_Comm comm, MPI_Comm machine_comm)
: is_parallel(false)
, comm_rank(MessagePassing::A_NULL_RANK)
, comm_nb_rank(0)
, stat(nullptr)
, trace_mng(nullptr)
, timer_mng(nullptr)
, thread_mng(nullptr)
, mpi_comm(comm)
, mpi_machine_comm(machine_comm)
, is_mpi_comm_owned(true)
, mpi_lock(nullptr)
{
  ::MPI_Comm_rank(comm,&comm_rank);
  ::MPI_Comm_size(comm,&comm_nb_rank);

  m_dispatchers_ref = createRef<MP::Dispatchers>();
  MP::Mpi::MpiMessagePassingMng::BuildInfo bi(comm_rank,comm_nb_rank,m_dispatchers_ref.get(),mpi_comm);

  m_message_passing_mng_ref = createRef<MP::Mpi::MpiMessagePassingMng>(bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Communicateur spécifique créé via MPI_Dist_graph_create_adjacent.
 */
class VariableSynchronizerMpiCommunicator
: public IVariableSynchronizerMpiCommunicator
{
 public:
  explicit VariableSynchronizerMpiCommunicator(MpiParallelMng* pm)
  : m_mpi_parallel_mng(pm){}
  ~VariableSynchronizerMpiCommunicator() override
  {
    _checkFreeCommunicator();
  }
  MPI_Comm communicator() const override
  {
    return m_topology_communicator;
  }
  void compute(VariableSynchronizer* var_syncer) override
  {
    Int32ConstArrayView comm_ranks = var_syncer->communicatingRanks();
    const Int32 nb_message = comm_ranks.size();

    MpiParallelMng* pm = m_mpi_parallel_mng;

    MPI_Comm old_comm = pm->communicator();

    UniqueArray<int> destinations(nb_message);
    for( Integer i=0; i<nb_message; ++i ){
      destinations[i] = comm_ranks[i];
    }

    _checkFreeCommunicator();

    int r = MPI_Dist_graph_create_adjacent(old_comm, nb_message, destinations.data(), MPI_UNWEIGHTED,
                                           nb_message, destinations.data(), MPI_UNWEIGHTED,
                                           MPI_INFO_NULL, 0, &m_topology_communicator);

    if (r!=MPI_SUCCESS)
      ARCANE_FATAL("Error '{0}' in MPI_Dist_graph_create",r);

    // Vérifie que l'ordre des rangs pour l'implémentation MPI est le même que celui qu'on a dans
    // le VariableSynchronizer.
    {
      int indegree = 0;
      int outdegree = 0;
      int weighted = 0;
      MPI_Dist_graph_neighbors_count(m_topology_communicator,&indegree,&outdegree,&weighted);

      if (indegree!=nb_message)
        ARCANE_FATAL("Bad value '{0}' for 'indegree' (expected={1})",indegree,nb_message);
      if (outdegree!=nb_message)
        ARCANE_FATAL("Bad value '{0}' for 'outdegree' (expected={1})",outdegree,nb_message);

      UniqueArray<int> srcs(indegree);
      UniqueArray<int> dsts(outdegree);

      MPI_Dist_graph_neighbors(m_topology_communicator,indegree,srcs.data(),MPI_UNWEIGHTED,outdegree,dsts.data(),MPI_UNWEIGHTED);

      for(int k=0; k<outdegree; ++k){
        int x = dsts[k];
        if (x!=comm_ranks[k])
          ARCANE_FATAL("Invalid destination rank order k={0} v={1} expected={2}",k,x,comm_ranks[k]);
      }

      for(int k=0; k<indegree; ++k ){
        int x = srcs[k];
        if (x!=comm_ranks[k])
          ARCANE_FATAL("Invalid source rank order k={0} v={1} expected={2}",k,x,comm_ranks[k]);
      }
    }
  }

 private:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
  MPI_Comm m_topology_communicator = MPI_COMM_NULL;

 private:

  void _checkFreeCommunicator()
  {
    if (m_topology_communicator!=MPI_COMM_NULL)
      MPI_Comm_free(&m_topology_communicator);
    m_topology_communicator = MPI_COMM_NULL;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Synchronizer spécifique MPI.
 *
 * Cette classe surcharge VariableSynchronizer::compute() pour calculer
 * un communicateur spécifique.
 */
class MpiVariableSynchronizer
: public VariableSynchronizer
{
 public:
  MpiVariableSynchronizer(IParallelMng* pm,const ItemGroup& group,
                          Ref<IDataSynchronizeImplementationFactory> implementation_factory,
                          Ref<IVariableSynchronizerMpiCommunicator> topology_info)
  : VariableSynchronizer(pm,group,implementation_factory)
  , m_topology_info(topology_info)
  {
  }
 public:
  void compute() override
  {
    VariableSynchronizer::compute();
    // Si non nul, calcule la topologie
    if (m_topology_info.get())
      m_topology_info->compute(this);
  }
 private:
  Ref<IVariableSynchronizerMpiCommunicator> m_topology_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiParallelMngUtilsFactory
: public ParallelMngUtilsFactoryBase
{
 public:
  MpiParallelMngUtilsFactory()
    : m_synchronizer_version(2)
  {
    if (platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_VERSION")=="1")
      m_synchronizer_version = 1;
    if (platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_VERSION")=="2")
      m_synchronizer_version = 2;
    if (platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_VERSION")=="3")
      m_synchronizer_version = 3;
    if (platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_VERSION")=="4"){
      m_synchronizer_version = 4;
      String v = platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_BLOCK_SIZE");
      if (!v.null()){
        Int32 block_size = 0;
        if (!builtInGetValue(block_size,v))
          m_synchronize_block_size = block_size;
        m_synchronize_block_size = std::clamp(m_synchronize_block_size,0,1000000000);
      }
      v = platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_NB_SEQUENCE");
      if (!v.null()){
        Int32 nb_sequence = 0;
        if (!builtInGetValue(nb_sequence,v))
          m_synchronize_nb_sequence = nb_sequence;
        m_synchronize_nb_sequence = std::clamp(m_synchronize_nb_sequence,1,1024*1024);
      }
    }
    if (platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_VERSION")=="5")
      m_synchronizer_version = 5;
    if (platform::getEnvironmentVariable("ARCANE_SYNCHRONIZE_VERSION")=="6")
      m_synchronizer_version = 6;
  }
 public:

  Ref<IVariableSynchronizer> createSynchronizer(IParallelMng* pm,IItemFamily* family) override
  {
    return _createSynchronizer(pm,family->allItems());
  }

  Ref<IVariableSynchronizer> createSynchronizer(IParallelMng* pm,const ItemGroup& group) override
  {
    return _createSynchronizer(pm,group);
  }

 private:

  Ref<IVariableSynchronizer> _createSynchronizer(IParallelMng* pm,const ItemGroup& group)
  {
    Ref<IVariableSynchronizerMpiCommunicator> topology_info;
    MpiParallelMng* mpi_pm = ARCANE_CHECK_POINTER(dynamic_cast<MpiParallelMng*>(pm));
    ITraceMng* tm = pm->traceMng();
    Ref<IDataSynchronizeImplementationFactory> generic_factory;
    // N'affiche les informations que pour le groupe de toutes les mailles pour éviter d'afficher
    // plusieurs fois le même message.
    bool do_print = (group.isAllItems() && group.itemKind()==IK_Cell);
    if (m_synchronizer_version == 2){
      if (do_print)
        tm->info() << "Using MpiSynchronizer V2";
      generic_factory = arcaneCreateMpiVariableSynchronizerFactory(mpi_pm);
    }
    else if (m_synchronizer_version == 3 ){
      if (do_print)
        tm->info() << "Using MpiSynchronizer V3";
      generic_factory = arcaneCreateMpiDirectSendrecvVariableSynchronizerFactory(mpi_pm);
    }
    else if (m_synchronizer_version == 4){
      if (do_print)
        tm->info() << "Using MpiSynchronizer V4 block_size=" << m_synchronize_block_size
                   << " nb_sequence=" << m_synchronize_nb_sequence;
      generic_factory = arcaneCreateMpiBlockVariableSynchronizerFactory(mpi_pm,m_synchronize_block_size,m_synchronize_nb_sequence);
    }
    else if (m_synchronizer_version == 5){
      if (do_print)
        tm->info() << "Using MpiSynchronizer V5";
      topology_info = createRef<VariableSynchronizerMpiCommunicator>(mpi_pm);
#if defined(ARCANE_HAS_MPI_NEIGHBOR)
      generic_factory = arcaneCreateMpiNeighborVariableSynchronizerFactory(mpi_pm,topology_info);
#else
      throw NotSupportedException(A_FUNCINFO,"Synchronize implementation V5 is not supported with this version of MPI");
#endif
    }
#if defined(ARCANE_HAS_PACKAGE_NCCL)
    else if (m_synchronizer_version == 6){
      if (do_print)
        tm->info() << "Using NCCLSynchronizer";
      generic_factory = arcaneCreateNCCLVariableSynchronizerFactory(mpi_pm);
    }
#endif
    else{
      if (do_print)
        tm->info() << "Using MpiSynchronizer V1";
      generic_factory = arcaneCreateMpiLegacyVariableSynchronizerFactory(mpi_pm);
    }
    if (!generic_factory.get())
      ARCANE_FATAL("No factory created");
    return createRef<MpiVariableSynchronizer>(pm,group,generic_factory,topology_info);
  }

 private:

  Integer m_synchronizer_version = 1;
  Int32 m_synchronize_block_size = 32000;
  Int32 m_synchronize_nb_sequence = 1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiParallelMng::Impl
: public ParallelMngInternal
{
 public:

  explicit Impl(MpiParallelMng* pm)
  : ParallelMngInternal(pm)
  , m_parallel_mng(pm)
  {}

  ~Impl() override = default;

 public:

  Ref<IMachineMemoryWindowBaseInternal> createMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(m_parallel_mng->adapter()->windowCreator(m_parallel_mng->machineCommunicator())->createWindow(sizeof_segment, sizeof_type));
  }

  Ref<IDynamicMachineMemoryWindowBaseInternal> createDynamicMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(m_parallel_mng->adapter()->windowCreator(m_parallel_mng->machineCommunicator())->createDynamicWindow(sizeof_segment, sizeof_type));
  }

 private:

  MpiParallelMng* m_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiParallelMng::
MpiParallelMng(const MpiParallelMngBuildInfo& bi)
: ParallelMngDispatcher(ParallelMngDispatcherBuildInfo(bi.dispatchersRef(),bi.messagePassingMngRef()))
, m_trace(bi.trace_mng)
, m_thread_mng(bi.thread_mng)
, m_world_parallel_mng(bi.world_parallel_mng)
, m_timer_mng(bi.timer_mng)
, m_replication(new ParallelReplication())
, m_is_parallel(bi.is_parallel)
, m_comm_rank(bi.commRank())
, m_comm_size(bi.commSize())
, m_stat(bi.stat)
, m_communicator(bi.mpiComm())
, m_machine_communicator(bi.mpiMachineComm())
, m_is_communicator_owned(bi.is_mpi_comm_owned)
, m_mpi_lock(bi.mpi_lock)
, m_non_blocking_collective(nullptr)
, m_utils_factory(createRef<MpiParallelMngUtilsFactory>())
, m_parallel_mng_internal(new Impl(this))
{
  if (!m_world_parallel_mng){
    m_trace->debug()<<"[MpiParallelMng] No m_world_parallel_mng found, reverting to ourselves!";
    m_world_parallel_mng = this;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiParallelMng::
~MpiParallelMng()
{
  delete m_parallel_mng_internal;
  delete m_non_blocking_collective;
  m_sequential_parallel_mng.reset();
  if (m_is_communicator_owned){
    MpiLock::Section ls(m_mpi_lock);
    MPI_Comm_free(&m_communicator);
    MPI_Comm_free(&m_machine_communicator);
  }
  delete m_replication;
  delete m_io_mng;
  if (m_is_timer_owned)
    delete m_timer_mng;
  arcaneCallFunctionAndTerminateIfThrow([&]() { m_adapter->destroy(); });
  delete m_datatype_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Classe pour créer les différents dispatchers
class DispatchCreator
{
 public:
  DispatchCreator(ITraceMng* tm,IMessagePassingMng* mpm,MpiAdapter* adapter,MpiDatatypeList* datatype_list)
  : m_tm(tm), m_mpm(mpm), m_adapter(adapter), m_datatype_list(datatype_list){}
 public:
  template<typename DataType> MpiParallelDispatchT<DataType>*
  create()
  {
    MpiDatatype* dt = m_datatype_list->datatype(DataType());
    return new MpiParallelDispatchT<DataType>(m_tm,m_mpm,m_adapter,dt);
  }

  ITraceMng* m_tm;
  IMessagePassingMng* m_mpm;
  MpiAdapter* m_adapter;
  MpiDatatypeList* m_datatype_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ControlDispatcherDecorator
: public ParallelMngDispatcher::DefaultControlDispatcher
{
 public:

  ControlDispatcherDecorator(IParallelMng* pm, MpiAdapter* adapter)
  : ParallelMngDispatcher::DefaultControlDispatcher(pm), m_adapter(adapter) {}

  IMessagePassingMng* commSplit(bool keep) override
  {
    return m_adapter->commSplit(keep);
  }
  MP::IProfiler* profiler() const override { return m_adapter->profiler(); }
  void setProfiler(MP::IProfiler* p) override { m_adapter->setProfiler(p); }

 private:
  MpiAdapter* m_adapter;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
build()
{
  ITraceMng* tm = traceMng();
  if (!m_timer_mng){
    m_timer_mng = new MpiTimerMng(tm);
    m_is_timer_owned = true;
  }

  // Créé le gestionnaire séquentiel associé.
  {
    SequentialParallelMngBuildInfo bi(timerMng(),worldParallelMng());
    bi.setTraceMng(traceMng());
    bi.setCommunicator(communicator());
    bi.setThreadMng(threadMng());
    m_sequential_parallel_mng = arcaneCreateSequentialParallelMngRef(bi);
  }

  // Indique que les reduces doivent être fait dans l'ordre des processeurs
  // afin de garantir une exécution déterministe
  bool is_ordered_reduce = false;
  if (platform::getEnvironmentVariable("ARCANE_ORDERED_REDUCE")=="TRUE")
    is_ordered_reduce = true;
  m_datatype_list = new MpiDatatypeList(is_ordered_reduce);

  ARCANE_CHECK_POINTER(m_stat);

  MpiAdapter* adapter = new MpiAdapter(m_trace,m_stat->toArccoreStat(),m_communicator,m_mpi_lock);
  m_adapter = adapter;
  auto mpm = _messagePassingMng();

  // NOTE: cette instance sera détruite par le ParallelMngDispatcher
  auto* control_dispatcher = new ControlDispatcherDecorator(this,m_adapter);
  _setControlDispatcher(control_dispatcher);

  // NOTE: cette instance sera détruite par le ParallelMngDispatcher
  auto* serialize_dispatcher = new MpiSerializeDispatcher(m_adapter, mpm);
  m_mpi_serialize_dispatcher = serialize_dispatcher;
  _setSerializeDispatcher(serialize_dispatcher);

  DispatchCreator creator(m_trace,mpm,m_adapter,m_datatype_list);
  this->createDispatchers(creator);

  m_io_mng = arcaneCreateIOMng(this);

  m_non_blocking_collective = new MpiParallelNonBlockingCollective(tm,this,adapter);
  m_non_blocking_collective->build();
  if (m_mpi_lock)
    m_trace->info() << "Using mpi with locks.";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
initialize()
{
  Trace::Setter mci(m_trace,"Mpi");
  if (m_is_initialized){
    m_trace->warning() << "MpiParallelMng already initialized";
    return;
  }
	
  m_trace->info() << "Initialisation de MpiParallelMng";
  m_sequential_parallel_mng->initialize();

  m_adapter->setTimeMetricCollector(timeMetricCollector());

  m_is_initialized = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
sendSerializer(ISerializer* s,Int32 rank)
{
  Trace::Setter mci(m_trace,"Mpi");
  Timer::Phase tphase(timeStats(),TP_Communication);
  MessageTag mpi_tag = BasicSerializeMessage::defaultTag();
  m_mpi_serialize_dispatcher->legacySendSerializer(s,{MessageRank(rank),mpi_tag,Blocking});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* MpiParallelMng::
createSendSerializer(Int32 rank)
{
  return m_utils_factory->createSendSerializeMessage(this, rank)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiParallelMng::
sendSerializer(ISerializer* s,Int32 rank,[[maybe_unused]] ByteArray& bytes)
{
  Trace::Setter mci(m_trace,"Mpi");
  Timer::Phase tphase(timeStats(),TP_Communication);
  MessageTag mpi_tag = BasicSerializeMessage::defaultTag();
  return m_mpi_serialize_dispatcher->legacySendSerializer(s,{MessageRank(rank),mpi_tag,NonBlocking});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
broadcastSerializer(ISerializer* values,Int32 rank)
{
  Timer::Phase tphase(timeStats(),TP_Communication);
  m_mpi_serialize_dispatcher->broadcastSerializer(values,MessageRank(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
recvSerializer(ISerializer* values,Int32 rank)
{
  Trace::Setter mci(m_trace,"Mpi");
  Timer::Phase tphase(timeStats(),TP_Communication);
  MessageTag mpi_tag = BasicSerializeMessage::defaultTag();
  m_mpi_serialize_dispatcher->legacyReceiveSerializer(values,MessageRank(rank),mpi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* MpiParallelMng::
createReceiveSerializer(Int32 rank)
{
  return m_utils_factory->createReceiveSerializeMessage(this, rank)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto MpiParallelMng::
probe(const PointToPointMessageInfo& message) -> MessageId
{
  return m_adapter->probeMessage(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto MpiParallelMng::
legacyProbe(const PointToPointMessageInfo& message) -> MessageSourceInfo
{
  return m_adapter->legacyProbeMessage(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiParallelMng::
sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message)
{
  return m_mpi_serialize_dispatcher->sendSerializer(s,message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiParallelMng::
receiveSerializer(ISerializer* s,const PointToPointMessageInfo& message)
{
  return m_mpi_serialize_dispatcher->receiveSerializer(s,message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
freeRequests(ArrayView<Parallel::Request> requests)
{
  for( Integer i=0, is=requests.size(); i<is; ++i )
    m_adapter->freeRequest(requests[i]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
_checkFinishedSubRequests()
{
  m_mpi_serialize_dispatcher->checkFinishedSubRequests();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> MpiParallelMng::
sequentialParallelMngRef()
{
  return m_sequential_parallel_mng;
}

IParallelMng* MpiParallelMng::
sequentialParallelMng()
{
  return m_sequential_parallel_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
printStats()
{
  if (m_stat)
    m_stat->print(m_trace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
barrier()
{
  traceMng()->flush();
  m_adapter->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
waitAllRequests(ArrayView<Request> requests)
{
  m_adapter->waitAllRequests(requests);
  _checkFinishedSubRequests();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueArray<Integer> MpiParallelMng::
waitSomeRequests(ArrayView<Request> requests)
{
  return _waitSomeRequests(requests, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueArray<Integer> MpiParallelMng::
testSomeRequests(ArrayView<Request> requests)
{
  return _waitSomeRequests(requests, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueArray<Integer> MpiParallelMng::
_waitSomeRequests(ArrayView<Request> requests, bool is_non_blocking)
{
  UniqueArray<Integer> results;
  UniqueArray<bool> done_indexes(requests.size());

  m_adapter->waitSomeRequests(requests, done_indexes, is_non_blocking);
  for (int i = 0 ; i < requests.size() ; i++) {
    if (done_indexes[i])
      results.add(i);
  }
  return results;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessageList* MpiParallelMng::
_createSerializeMessageList()
{
  return new MP::internal::SerializeMessageList(messagePassingMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IGetVariablesValuesParallelOperation* MpiParallelMng::
createGetVariablesValuesOperation()
{
  return m_utils_factory->createGetVariablesValuesOperation(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITransferValuesParallelOperation* MpiParallelMng::
createTransferValuesOperation()
{
  return m_utils_factory->createTransferValuesOperation(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelExchanger* MpiParallelMng::
createExchanger()
{
  return m_utils_factory->createExchanger(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* MpiParallelMng::
createSynchronizer(IItemFamily* family)
{
  return m_utils_factory->createSynchronizer(this,family)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* MpiParallelMng::
createSynchronizer(const ItemGroup& group)
{
  return m_utils_factory->createSynchronizer(this,group)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelTopology* MpiParallelMng::
createTopology()
{
  return m_utils_factory->createTopology(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelReplication* MpiParallelMng::
replication() const
{
  return m_replication;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelMng::
setReplication(IParallelReplication* v)
{
  delete m_replication;
  m_replication = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* MpiParallelMng::
_createSubParallelMng(MPI_Comm sub_communicator)
{
  // Si nul, ce rang ne fait pas partie du sous-communicateur
  if (sub_communicator==MPI_COMM_NULL)
    return nullptr;

  int sub_rank = -1;
  MPI_Comm_rank(sub_communicator,&sub_rank);

  MPI_Comm sub_machine_communicator = MPI_COMM_NULL;
  MPI_Comm_split_type(sub_communicator, MPI_COMM_TYPE_SHARED, sub_rank, MPI_INFO_NULL, &sub_machine_communicator);

  MpiParallelMngBuildInfo bi(sub_communicator, sub_machine_communicator);
  bi.is_parallel = isParallel();
  bi.stat = m_stat;
  bi.timer_mng = m_timer_mng;
  bi.thread_mng = m_thread_mng;
  bi.trace_mng = m_trace;
  bi.world_parallel_mng = m_world_parallel_mng;
  bi.mpi_lock = m_mpi_lock;

  IParallelMng* sub_pm = new MpiParallelMng(bi);
  sub_pm->build();
  return sub_pm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> MpiParallelMng::
_createSubParallelMngRef(Int32 color, Int32 key)
{
  if (color < 0)
    color = MPI_UNDEFINED;
  MPI_Comm sub_communicator = MPI_COMM_NULL;
  MPI_Comm_split(m_communicator, color, key, &sub_communicator);
  IParallelMng* sub_pm = _createSubParallelMng(sub_communicator);
  return makeRef(sub_pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* MpiParallelMng::
_createSubParallelMng(Int32ConstArrayView kept_ranks)
{
  MPI_Group mpi_group = MPI_GROUP_NULL;
  MPI_Comm_group(m_communicator, &mpi_group);
  Integer nb_sub_rank = kept_ranks.size();
  UniqueArray<int> mpi_kept_ranks(nb_sub_rank);
  for (Integer i = 0; i < nb_sub_rank; ++i)
    mpi_kept_ranks[i] = (int)kept_ranks[i];

  MPI_Group final_group = MPI_GROUP_NULL;
  MPI_Group_incl(mpi_group, nb_sub_rank, mpi_kept_ranks.data(), &final_group);
  MPI_Comm sub_communicator = MPI_COMM_NULL;

  MPI_Comm_create(m_communicator, final_group, &sub_communicator);
  MPI_Group_free(&final_group);
  return _createSubParallelMng(sub_communicator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation de MpiRequestList pour MpiParallelMng.
 *
 * Cette classe fait juste en sorte d'appeler _checkFinishedSubRequests();
 * après les wait. Elle ne sera plus utile lorsqu'on utilisera l'implémentation
 * 'SerializeMessageList' de message_passing.
 */
class MpiParallelMng::RequestList
: public MpiRequestList
{
  using Base = MpiRequestList;
 public:
  explicit RequestList(MpiParallelMng* pm)
  : Base(pm->m_adapter), m_parallel_mng(pm){}
 public:
  void _wait(Parallel::eWaitType wait_type) override
  {
    Base::_wait(wait_type);
    m_parallel_mng->_checkFinishedSubRequests();
  };
 private:
  MpiParallelMng* m_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IRequestList> MpiParallelMng::
createRequestListRef()
{
  return createRef<RequestList>(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMngUtilsFactory> MpiParallelMng::
_internalUtilsFactory() const
{
  return m_utils_factory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MpiParallelMng::
_isAcceleratorAware() const
{
  return arcaneIsAcceleratorAwareMPI();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
