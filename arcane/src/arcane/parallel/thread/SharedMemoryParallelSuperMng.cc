// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryParallelSuperMng.cc                             (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de messages utilisant uniquement la mémoire partagée.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/SharedMemoryParallelSuperMng.h"

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/IThreadMng.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IThreadImplementation.h"
#include "arcane/utils/Mutex.h"

#include "arcane/parallel/IStat.h"

#include "arcane/parallel/thread/SharedMemoryParallelMng.h"
#include "arcane/parallel/thread/SharedMemoryParallelDispatch.h"
#include "arcane/parallel/thread/SharedMemoryMessageQueue.h"
#include "arcane/parallel/thread/internal/SharedMemoryThreadMng.h"
#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBaseInternalCreator.h"

#include "arcane/core/FactoryService.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/ParallelSuperMngDispatcher.h"
#include "arcane/core/ApplicationBuildInfo.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/ServiceBuilder.h"

#include "arcane/core/IMainFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur des informations du gestionnaire de message en mémoire partagée.
 */
class SharedMemoryParallelMngContainer
: public ParallelMngContainerBase
{
 public:
  SharedMemoryParallelMngContainer(IApplication* app,Int32 nb_local_rank,
                                   MP::Communicator mpi_comm,
                                   IParallelMngContainerFactory* factory);
  ~SharedMemoryParallelMngContainer() override;

 public:

  void build();
  Ref<IParallelMng> _createParallelMng(Int32 local_rank,ITraceMng* tm) override;

 public:

  IApplication* m_application; //!< Gestionnaire principal
  Int32 m_nb_local_rank;
  SharedMemoryThreadMng* m_thread_mng;
  ISharedMemoryMessageQueue* m_message_queue = nullptr;
  Mutex* m_internal_create_mutex = nullptr;
  IThreadBarrier* m_thread_barrier = nullptr;
  SharedMemoryAllDispatcher* m_all_dispatchers = nullptr;
  IParallelMngContainerFactory* m_sub_factory_builder = nullptr;
  SharedMemoryMachineMemoryWindowBaseInternalCreator* m_window_creator = nullptr;

 private:

  MP::Communicator m_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryParallelMngContainer::
SharedMemoryParallelMngContainer(IApplication* app,Int32 nb_local_rank,
                                 MP::Communicator mpi_comm,
                                 IParallelMngContainerFactory* factory)
: m_application(app), m_nb_local_rank(nb_local_rank)
, m_thread_mng(new SharedMemoryThreadMng())
, m_sub_factory_builder(factory)
, m_communicator(mpi_comm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryParallelMngContainer::
~SharedMemoryParallelMngContainer()
{
  if (m_thread_barrier)
    m_thread_barrier->destroy();
  delete m_message_queue;
  delete m_thread_mng;
  delete m_all_dispatchers;
  delete m_internal_create_mutex;
  delete m_window_creator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMngContainer::
build()
{
  m_message_queue = new SharedMemoryMessageQueue();
  m_message_queue->init(m_nb_local_rank);

  m_thread_barrier = platform::getThreadImplementationService()->createBarrier();
  m_thread_barrier->init(m_nb_local_rank);

  m_all_dispatchers = new SharedMemoryAllDispatcher();
  m_all_dispatchers->resize(m_nb_local_rank);

  m_internal_create_mutex = new Mutex();

  m_window_creator = new SharedMemoryMachineMemoryWindowBaseInternalCreator(m_nb_local_rank, m_thread_barrier);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> SharedMemoryParallelMngContainer::
_createParallelMng(Int32 local_rank,ITraceMng* tm)
{
  if (local_rank<0 || local_rank>=m_nb_local_rank)
    ARCANE_THROW(ArgumentException,"Bad value '{0}' for local_rank (max={1})",
                 local_rank,m_nb_local_rank);

  // Cette méthode n'est pas réentrante.
  Mutex::ScopedLock sl(m_internal_create_mutex);

  IParallelSuperMng* sm = m_application->sequentialParallelSuperMng();
  SharedMemoryParallelMngBuildInfo build_info;
  build_info.rank = local_rank;
  build_info.nb_rank = m_nb_local_rank;
  build_info.trace_mng = tm;
  build_info.thread_mng = m_thread_mng;
  build_info.sequential_parallel_mng = sm->internalCreateWorldParallelMng(0);
  build_info.world_parallel_mng = nullptr;
  build_info.message_queue = m_message_queue;
  build_info.thread_barrier = m_thread_barrier;
  build_info.all_dispatchers = m_all_dispatchers;
  build_info.sub_builder_factory = m_sub_factory_builder;
  build_info.container = makeRef<IParallelMngContainer>(this);
  build_info.window_creator = m_window_creator;
  // Seul le rang 0 positionne l'éventuel communicateur sinon tous les PE
  // vont se retrouver avec le même rang MPI
  if (local_rank==0)
    build_info.communicator = m_communicator;

  IParallelMng* pm = new SharedMemoryParallelMng(build_info);
  pm->build();

  return makeRef(pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryParallelMngContainerFactory
: public AbstractService
, public IParallelMngContainerFactory
{
 public:
  SharedMemoryParallelMngContainerFactory(const ServiceBuildInfo& sbi)
  : AbstractService(sbi), m_application(sbi.application()){}
 public:
  Ref<IParallelMngContainer> _createParallelMngBuilder(Int32 nb_rank, MP::Communicator comm, MP::Communicator machine_comm) override
  {
    ARCANE_UNUSED(machine_comm);
    auto x = new SharedMemoryParallelMngContainer(m_application,nb_rank,comm,this);
    x->build();
    return makeRef<IParallelMngContainer>(x);
  }
 private:
  IApplication* m_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SharedMemoryParallelMngContainerFactory,
                        ServiceProperty("SharedMemoryParallelMngContainerFactory",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelMngContainerFactory));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryParallelSuperMng::
SharedMemoryParallelSuperMng(const ServiceBuildInfo& sbi)
: SharedMemoryParallelSuperMng(sbi,MP::Communicator(),false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryParallelSuperMng::
SharedMemoryParallelSuperMng(const ServiceBuildInfo& sbi,MP::Communicator comm,
                             bool has_mpi_init)
: m_application(sbi.application())
, m_stat(nullptr)
, m_is_parallel(false)
, m_communicator(comm)
{
  m_stat = Parallel::createDefaultStat();
  m_has_mpi_init = has_mpi_init;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryParallelSuperMng::
~SharedMemoryParallelSuperMng()
{
  delete m_stat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelSuperMng::
initialize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelSuperMng::
build()
{
  if (!arcaneHasThread())
    ARCANE_FATAL("Can not create SharedMemoryParallelSuperMng because threads are disabled");

  // Si on a été initialisé avec MPI, alors les requêtes nulles et le communicateur
  // nul ont déjà été positionnés.
  if (!m_has_mpi_init){
    Request::setNullRequest(Request(0,nullptr,0));
    Parallel::Communicator::setNullCommunicator(Parallel::Communicator((void*)nullptr));
  }

  m_is_parallel  = true;
  Int32 n = m_application->applicationBuildInfo().nbSharedMemorySubDomain();
  if (n==0)
    ARCANE_FATAL("Number of shared memory sub-domains is not defined");

  ITraceMng* app_tm = m_application->traceMng();
  app_tm->info() << "SharedMemoryParallelSuperMng: nb_local_sub_domain=" << n;
  app_tm->info() << "SharedMemoryParallelSuperMng: mpi_communicator=" << getMPICommunicator();
  ServiceBuilder<IParallelMngContainerFactory> sb(m_application);
  String service_name = "SharedMemoryParallelMngContainerFactory";
  m_builder_factory = sb.createReference(service_name);
  Ref<IParallelMngContainer> x = m_builder_factory->_createParallelMngBuilder(n, communicator(), communicator());
  m_main_builder = x;
  m_container = dynamic_cast<SharedMemoryParallelMngContainer*>(x.get());
  ARCANE_CHECK_POINTER(m_container);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> SharedMemoryParallelSuperMng::
internalCreateWorldParallelMng(Int32 local_rank)
{
  Int32 max_rank = nbLocalSubDomain();
  if (local_rank<0 || local_rank>=max_rank)
    ARCANE_THROW(ArgumentException,"Bad value '{0}' for local_rank (max={1})",
                 local_rank,max_rank);

  ITraceMng* tm = nullptr;
  ITraceMng* app_tm = m_application->traceMng();
  if (local_rank==0){
    // Le premier sous-domaine créé utilise le traceMng() par défaut.
    tm = app_tm;
  }
  else{
    tm = m_application->createAndInitializeTraceMng(app_tm,String::fromNumber(local_rank));
  }

  Ref<IParallelMng> pm = m_container->_createParallelMng(local_rank,tm);
  return pm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelSuperMng::
tryAbort()
{
  m_application->traceMng()->flush();
  ::abort();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelSuperMng::
broadcast(ByteArrayView send_buf,Int32 process_id)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(process_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelSuperMng::
broadcast(Int32ArrayView send_buf,Int32 process_id)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(process_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelSuperMng::
broadcast(Int64ArrayView send_buf,Int32 process_id)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(process_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelSuperMng::
broadcast(RealArrayView send_buf,Int32 process_id)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(process_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IThreadMng* SharedMemoryParallelSuperMng::
threadMng() const
{
  return m_container->m_thread_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 SharedMemoryParallelSuperMng::
nbLocalSubDomain()
{
  return m_container->m_nb_local_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SharedMemoryParallelSuperMng,
                        ServiceProperty("SharedMemoryParallelSuperMng",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelSuperMng));

// Ancien nom
ARCANE_REGISTER_SERVICE(SharedMemoryParallelSuperMng,
                        ServiceProperty("ThreadParallelSuperMng",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelSuperMng));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
