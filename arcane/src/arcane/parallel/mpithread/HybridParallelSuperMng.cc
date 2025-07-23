// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridParallelSuperMng.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI et mémoire partagée.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/NullThreadMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/CommandLineArguments.h"

#include "arcane/parallel/IStat.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiParallelDispatch.h"
#include "arcane/parallel/mpi/MpiErrorHandler.h"

#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arccore/message_passing_mpi/internal/MpiLock.h"

#include "arcane/parallel/thread/SharedMemoryMessageQueue.h"
#include "arcane/parallel/thread/SharedMemoryParallelMng.h"
#include "arcane/parallel/thread/SharedMemoryParallelSuperMng.h"
#include "arcane/parallel/thread/internal/SharedMemoryThreadMng.h"

#include "arcane/parallel/mpithread/HybridParallelMng.h"
#include "arcane/parallel/mpithread/HybridParallelDispatch.h"
#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBaseInternalCreator.h"

#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/ParallelSuperMngDispatcher.h"
#include "arcane/core/ApplicationBuildInfo.h"
#include "arcane/core/ServiceBuilder.h"

#include "arcane/core/IMainFactory.h"

#include "arccore/base/ReferenceCounterImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur des informations du gestionnaire de message en mode hybride
 */
class HybridParallelMngContainer
: public ParallelMngContainerBase
{
 public:
  HybridParallelMngContainer(IApplication* app,Int32 nb_local_rank,
                             MP::Communicator mpi_comm, IParallelMngContainerFactory* factory,
                             Parallel::IStat* stat,MpiLock* mpi_lock);
  ~HybridParallelMngContainer() override;


 public:

  void build();
  Ref<IParallelMng> _createParallelMng(Int32 local_rank,ITraceMng* tm) override;

 public:

  IApplication* m_application; //!< Gestionnaire principal
  Parallel::IStat* m_stat = nullptr; //! Statistiques
  IThreadMng* m_thread_mng = nullptr;
  MpiLock* m_mpi_lock = nullptr;
  ISharedMemoryMessageQueue* m_message_queue = nullptr;
  IThreadBarrier* m_thread_barrier = nullptr;
  Int32 m_local_nb_rank = -1;
  MpiThreadAllDispatcher* m_all_dispatchers = nullptr;
  // Cet objet est partagé par tous les HybridParallelMng.
  UniqueArray<HybridParallelMng*>* m_parallel_mng_list = nullptr;
  Mutex* m_internal_create_mutex = nullptr;
  IParallelMngContainerFactory* m_sub_builder_factory = nullptr;
  HybridMachineMemoryWindowBaseInternalCreator* m_window_creator = nullptr;

 private:
  MPI_Comm m_mpi_communicator; //!< Communicateur MPI
  Int32 m_mpi_comm_rank = -1; //!< Numéro du processeur actuel
  Int32 m_mpi_comm_size = -1; //!< Nombre de processeurs
 private:
  void _setMPICommunicator();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridParallelMngContainer::
HybridParallelMngContainer(IApplication* app,Int32 nb_local_rank,
                           MP::Communicator mpi_comm, IParallelMngContainerFactory* factory,
                           Parallel::IStat* stat,MpiLock* mpi_lock)
: m_application(app)
, m_stat(stat)
, m_thread_mng(new SharedMemoryThreadMng())
, m_mpi_lock(mpi_lock)
, m_local_nb_rank(nb_local_rank)
, m_parallel_mng_list(new UniqueArray<HybridParallelMng*>())
, m_sub_builder_factory(factory)
, m_mpi_communicator(mpi_comm)
{
  _setMPICommunicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridParallelMngContainer::
~HybridParallelMngContainer()
{
  // TODO: regarder s'il faut détruire le communicateur
  m_thread_barrier->destroy();
  delete m_message_queue;
  delete m_thread_mng;
  delete m_all_dispatchers;
  delete m_parallel_mng_list;
  delete m_internal_create_mutex;
  delete m_window_creator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMngContainer::
build()
{
  m_internal_create_mutex = new Mutex();

  m_all_dispatchers = new MpiThreadAllDispatcher();
  m_all_dispatchers->resize(m_local_nb_rank);

  m_parallel_mng_list->resize(m_local_nb_rank);
  m_parallel_mng_list->fill(nullptr);

  m_message_queue = new SharedMemoryMessageQueue();
  m_message_queue->init(m_local_nb_rank);

  m_thread_barrier = platform::getThreadImplementationService()->createBarrier();
  m_thread_barrier->init(m_local_nb_rank);

  m_window_creator = new HybridMachineMemoryWindowBaseInternalCreator(m_local_nb_rank, m_thread_barrier);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMngContainer::
_setMPICommunicator()
{
  MPI_Comm comm = static_cast<MPI_Comm>(m_mpi_communicator);

  if (comm==MPI_COMM_NULL)
    ARCANE_THROW(ArgumentException,"Null MPI Communicator");
  m_mpi_communicator = comm;

  int rank = 0;
  MPI_Comm_rank(m_mpi_communicator,&rank);
  int size = 0;
  MPI_Comm_size(m_mpi_communicator,&size);

  m_mpi_comm_rank = rank;
  m_mpi_comm_size = size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> HybridParallelMngContainer::
_createParallelMng(Int32 local_rank,ITraceMng* tm)
{
  if (local_rank<0 || local_rank>=m_local_nb_rank)
    ARCANE_THROW(ArgumentException,"Bad value '{0}' for local_rank (max={1})",
                 local_rank,m_local_nb_rank);

  // Cette méthode n'est pas réentrante.
  Mutex::ScopedLock sl(m_internal_create_mutex);

  Int32 nb_process = m_mpi_comm_size;
  bool is_parallel = nb_process > 1;

  // Le communicateur passé en argument reste notre propriété.
  MpiParallelMngBuildInfo bi(m_mpi_communicator);
  bi.is_parallel = is_parallel;
  bi.stat = m_stat;
  bi.trace_mng = tm;
  bi.thread_mng = m_thread_mng;
  bi.is_mpi_comm_owned = false;
  bi.mpi_lock = m_mpi_lock;

  if (m_mpi_lock)
    tm->info() << "MPI implementation need serialized threads : using locks";

  MpiParallelMng* mpi_pm = new MpiParallelMng(bi);

  mpi_pm->build();
  mpi_pm->initialize();
  mpi_pm->adapter()->enableDebugRequest(false);

  HybridParallelMngBuildInfo build_info;
  build_info.local_rank = local_rank;
  build_info.local_nb_rank = m_local_nb_rank;
  build_info.mpi_parallel_mng = mpi_pm;
  build_info.trace_mng = tm;
  build_info.thread_mng = m_thread_mng;
  build_info.message_queue = m_message_queue;
  build_info.thread_barrier = m_thread_barrier;
  build_info.parallel_mng_list = m_parallel_mng_list;
  build_info.all_dispatchers = m_all_dispatchers;
  build_info.sub_builder_factory = m_sub_builder_factory;
  build_info.container = makeRef<IParallelMngContainer>(this);
  build_info.window_creator = m_window_creator;

  // NOTE: Cette instance sera détruite par l'appelant de cette méthode
  HybridParallelMng* pm = new HybridParallelMng(build_info);
  pm->build();
  (*m_parallel_mng_list)[local_rank] = pm;

  return makeRef<IParallelMng>(pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridParallelMngContainerFactory
: public AbstractService
, public IParallelMngContainerFactory
{
 public:
  HybridParallelMngContainerFactory(const ServiceBuildInfo& sbi)
  : AbstractService(sbi), m_application(sbi.application()){}
 public:
  Ref<IParallelMngContainer>
  _createParallelMngBuilder(Int32 nb_rank,MP::Communicator mpi_communicator) override
  {
    auto x = new HybridParallelMngContainer(m_application,nb_rank,mpi_communicator,
                                            this,m_stat,m_mpi_lock);
    x->build();
    return makeRef<IParallelMngContainer>(x);
  }
 private:
  IApplication* m_application;
 public:
  MpiLock* m_mpi_lock = nullptr;
  Parallel::IStat* m_stat = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(HybridParallelMngContainerFactory,
                        ServiceProperty("HybridParallelMngContainerFactory",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelMngContainerFactory));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Superviseur du parallélisme utilisant MPI et Threads
 */
class HybridParallelSuperMng
: public ParallelSuperMngDispatcher
{
 public:

  explicit HybridParallelSuperMng(const ServiceBuildInfo& sbi);
  ~HybridParallelSuperMng() override;

  void initialize() override;
  void build() override;

  IApplication* application() const override { return m_application; }
  IThreadMng* threadMng() const override { return m_container->m_thread_mng; }
  bool isParallel() const override { return true; }
  Int32 commRank() const override { return m_mpi_comm_rank; }
  Int32 commSize() const override { return m_mpi_comm_size; }
  Int32 traceRank() const override { return m_mpi_comm_rank * m_container->m_local_nb_rank; }
  void* getMPICommunicator() override { return &m_mpi_communicator; }
  MP::Communicator communicator() const override { return m_communicator; }
  Ref<IParallelMng> internalCreateWorldParallelMng(Int32 local_rank) override;
  void tryAbort() override;
  bool isMasterIO() const override { return commRank()==0; }
  Integer masterIORank() const override { return 0; }
  Integer nbLocalSubDomain() override { return m_container->m_local_nb_rank; }
  void barrier() override;

 public:

 public:

  HybridParallelMngContainer* m_container = nullptr;
  Ref<IParallelMngContainerFactory> m_builder_factory;
  Ref<IParallelMngContainer> m_main_builder;

  IApplication* m_application; //!< Gestionnaire principal
  Parallel::IStat* m_stat; //! Statistiques
  Int32 m_mpi_comm_rank = -1; //!< Numéro du processeur actuel
  Int32 m_mpi_comm_size = -1; //!< Nombre de processeurs
  MPI_Comm m_mpi_communicator;
  MP::Communicator m_communicator;
  Int32 m_local_nb_rank = A_NULL_RANK;
  MpiLock* m_mpi_lock;
  MpiAdapter* m_mpi_adapter;
  MpiDatatypeList* m_datatype_list;
  MpiErrorHandler m_error_handler;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridParallelSuperMng::
HybridParallelSuperMng(const ServiceBuildInfo& sbi)
: m_application(sbi.application())
, m_stat(nullptr)
, m_mpi_comm_size(0)
, m_mpi_communicator(MPI_COMM_NULL)
, m_local_nb_rank(A_NULL_RANK)
, m_mpi_lock(nullptr)
, m_mpi_adapter(nullptr)
, m_datatype_list(nullptr)
{
  m_stat = Parallel::createDefaultStat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridParallelSuperMng::
~HybridParallelSuperMng()
{
  m_error_handler.removeHandler();
  _finalize();
  delete m_datatype_list;
  if (m_mpi_adapter)
    arcaneCallFunctionAndTerminateIfThrow([&]() { m_mpi_adapter->destroy(); });
  delete m_mpi_lock;
  delete m_stat;
  MPI_Barrier(m_mpi_communicator);
  MPI_Comm_free(&m_mpi_communicator);
  m_mpi_communicator = MPI_COMM_NULL;
  arcaneFinalizeMPI();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelSuperMng::
initialize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelSuperMng::
build()
{
  if (!arcaneHasThread())
    ARCANE_FATAL("Can not create HybridParallelSuperMng because threads are disabled");

  Request::setNullRequest(Request(0,nullptr,MPI_REQUEST_NULL));
  Communicator::setNullCommunicator(Communicator(MPI_COMM_NULL));

  int rank, size;
  int* argc = nullptr;
  char*** argv = nullptr;

  const ApplicationInfo& app_info = m_application->applicationInfo();
  const CommandLineArguments& cmd_line_args = app_info.commandLineArguments();
  argc = cmd_line_args.commandLineArgc();
  argv = cmd_line_args.commandLineArgv();
  
  bool need_serialize = false;

  // Essaie avec MPI_THREAD_MULTIPLE
  // ce qui permet d'appeler MPI depuis plusieurs threads en même temps
  // Si ce niveau n'existe pas, il faut au moins le niveau
  // MPI_THREAD_SERIALIZED ce qui permet d'appeler MPI depuis plusieurs
  // threads mais un seul à la fois. Il faut donc dans ce cas mettre
  // des verrous autour des appels MPI. Dans notre cas, les verrous
  // ne sont utiliser que pour les communications point à point car
  // pour les opérations collectives il n'y a qu'un seul thread qui les fait
  // à la fois.
  int thread_wanted = MPI_THREAD_MULTIPLE;
  int thread_provided = 0;

  Real start_time = platform::getRealTime();
  arcaneInitializeMPI(argc,argv,thread_wanted);
  Real end_time = platform::getRealTime();
  MPI_Query_thread(&thread_provided);

  if (thread_provided < MPI_THREAD_MULTIPLE) {
    if (thread_provided>=MPI_THREAD_SERIALIZED)
      need_serialize = true;
    else{
      // Le niveau de thread requis n'est pas disponible.
      // Lance un fatal mais seulement un seul processeur.
      int my_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
      if (my_rank!=0)
        ARCANE_FATAL("MPI thread level provided!=wanted ({0}!={1})",
                     thread_provided,thread_wanted);
    }
  }

  // decommenter pour forcer l'usage des verrous pour test.
  // need_serialize = true;

  if (need_serialize)
    m_mpi_lock = new MpiLock();

  MPI_Comm_dup(MPI_COMM_WORLD,&m_mpi_communicator);
  m_communicator = Communicator(m_mpi_communicator);
  MPI_Comm_rank(m_mpi_communicator,&rank);
  MPI_Comm_size(m_mpi_communicator,&size);

  m_mpi_comm_rank = rank;
  m_mpi_comm_size = size;
  IApplication* app = m_application;

  m_error_handler.registerHandler(m_mpi_communicator);

  Integer n = app->applicationBuildInfo().nbSharedMemorySubDomain();
  if (n==0)
    ARCANE_FATAL("Number of shared memory sub-domains is not defined");
  m_local_nb_rank = n;

  if (rank==0){
    ITraceMng* tm = app->traceMng();
    tm->info() << "MPI has non blocking collective";
    tm->info() << "MPI: sizeof(MPI_Count)=" << sizeof(MPI_Count);
    tm->info() << "MPI: is Cuda Aware?=" << arcaneIsCudaAwareMPI();
    tm->info() << "MPI: init_time (seconds)=" << (end_time-start_time);
  }

  /*cout << "ThreadSuperParallelMng: nb_sub_domain=" << m_local_nb_rank
       << " env=" << s
       << " mpi_rank=" << m_mpi_comm_rank << "/" << m_mpi_comm_size
       << " first_rank=" << m_current_rank << '\n';*/

  ITraceMng* tm = app->traceMng();
  m_datatype_list = new MpiDatatypeList(false);
  auto* adapter = new MpiAdapter(tm,m_stat->toArccoreStat(),m_communicator,nullptr);
  m_mpi_adapter = adapter;
  auto c = createBuiltInDispatcher<Byte>(tm,nullptr,m_mpi_adapter,m_datatype_list);
  auto i32 = createBuiltInDispatcher<Int32>(tm,nullptr,m_mpi_adapter,m_datatype_list);
  auto i64 = createBuiltInDispatcher<Int64>(tm,nullptr,m_mpi_adapter,m_datatype_list);
  auto r = createBuiltInDispatcher<Real>(tm,nullptr,m_mpi_adapter,m_datatype_list);
  _setDispatchers(c,i32,i64,r);

  ServiceBuilder<IParallelMngContainerFactory> sb(m_application);
  String service_name = "HybridParallelMngContainerFactory";
  m_builder_factory = sb.createReference(service_name);
  auto* true_builder = dynamic_cast<HybridParallelMngContainerFactory*>(m_builder_factory.get());
  ARCANE_CHECK_POINTER(true_builder);
  true_builder->m_stat = m_stat;
  true_builder->m_mpi_lock = m_mpi_lock;
  
  Ref<IParallelMngContainer> x = m_builder_factory->_createParallelMngBuilder(n,m_communicator);
  m_main_builder = x;
  m_container = dynamic_cast<HybridParallelMngContainer*>(x.get());
  ARCANE_CHECK_POINTER(m_container);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> HybridParallelSuperMng::
internalCreateWorldParallelMng(Int32 local_rank)
{
  if (local_rank<0 || local_rank>=m_local_nb_rank)
    throw ArgumentException(A_FUNCINFO,"Bad value for local_rank");
  
  Int32 current_global_rank = local_rank + m_local_nb_rank * commRank();
  ITraceMng* app_tm = m_application->traceMng();
  app_tm->info() << "Create SharedMemoryParallelMng rank=" << current_global_rank;

  ITraceMng* tm = nullptr;
  if (local_rank==0){
    // Le premier sous-domaine créé utilise le traceMng() par défaut.
    tm = app_tm;
  }
  else{
    tm = m_application->createAndInitializeTraceMng(app_tm,String::fromNumber(current_global_rank));
  }

  Ref<IParallelMng> pm = m_container->_createParallelMng(local_rank,tm);
  return pm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelSuperMng::
tryAbort()
{
  m_application->traceMng()->flush();
  MPI_Abort(m_communicator,2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelSuperMng::
barrier()
{
  MPI_Barrier(m_communicator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(HybridParallelSuperMng,
                        ServiceProperty("HybridParallelSuperMng",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelSuperMng));

// Ancien nom
ARCANE_REGISTER_SERVICE(HybridParallelSuperMng,
                        ServiceProperty("MpiThreadParallelSuperMng",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelSuperMng));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Superviseur en mémoire partagé avec initialisation MPI.
 *
 * Cela permet de se comporter comme en mode mémoire partagé mais avec
 * mais avec un communicateur MPI qui existe.
 */
class MpiSharedMemoryParallelSuperMng
: public SharedMemoryParallelSuperMng
{
 public:
  explicit MpiSharedMemoryParallelSuperMng(const ServiceBuildInfo& sbi)
  : SharedMemoryParallelSuperMng(sbi,Parallel::Communicator(MPI_COMM_WORLD),true)
  {
    Request::setNullRequest(Request(0,nullptr,MPI_REQUEST_NULL));
    Communicator::setNullCommunicator(Communicator(MPI_COMM_NULL));
  }

  ~MpiSharedMemoryParallelSuperMng() override
  {
    //arcaneFinalizeMPI();
  }

  void build() override
  {
    SharedMemoryParallelSuperMng::build();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MpiSharedMemoryParallelSuperMng,
                        ServiceProperty("MpiSharedMemoryParallelSuperMng",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelSuperMng));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
