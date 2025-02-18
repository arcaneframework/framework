// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelSuperMng.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/NullThreadMng.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/CommandLineArguments.h"

#include "arcane/parallel/IStat.h"

#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiParallelDispatch.h"
#include "arcane/parallel/mpi/MpiErrorHandler.h"

#include "arcane/FactoryService.h"
#include "arcane/IApplication.h"
#include "arcane/ParallelSuperMngDispatcher.h"

#include "arcane/impl/SequentialParallelSuperMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Superviseur du parallélisme utilisant MPI.
 */
class MpiParallelSuperMng
: public ParallelSuperMngDispatcher
{
 public:

  explicit MpiParallelSuperMng(const ServiceBuildInfo& sbi);
  ~MpiParallelSuperMng() override;

  void initialize() override;
  void build() override;

  IApplication* application() const override { return m_application; }
  IThreadMng* threadMng() const override { return m_thread_mng; }
  bool isParallel() const override { return m_is_parallel; }
  Int32 commRank() const override { return m_rank; }
  Int32 commSize() const override { return m_nb_rank; }
  Int32 traceRank() const override { return m_rank; }
  void* getMPICommunicator() override { return &m_mpi_main_communicator; }
  Parallel::Communicator communicator() const override { return m_main_communicator; }
  Ref<IParallelMng> internalCreateWorldParallelMng(Int32 local_rank) override;
  void tryAbort() override;
  bool isMasterIO() const override { return commRank()==0; }
  Integer masterIORank() const override { return 0; }
  Integer nbLocalSubDomain() override { return m_nb_local_sub_domain; }
  void barrier() override;

 public:

  static void initMPI(IApplication* app);

 public:

  IApplication* m_application; //!< Gestionnaire principal
  IThreadMng* m_thread_mng;
  Parallel::IStat* m_stat; //! Statistiques
  bool m_is_parallel;  //!< \a true si on est en mode parallèle
  Int32 m_rank; //!< Rang MPI dans le communicateur global de ce processus
  Int32 m_nb_rank; //!< Nombre de processus MPI dans le communicateur global
  Int32 m_nb_local_sub_domain; //!< Nombre de sous-domaines locaux
  MPI_Comm m_mpi_main_communicator; //!< Communicateur MPI
  MP::Communicator m_main_communicator; //!< Communicateur MPI
  MpiErrorHandler m_error_handler;
  MpiAdapter* m_adapter;
  MpiDatatypeList* m_datatype_list;

 private:
  
  // Handler d'erreur
  static void _ErrorHandler(MPI_Comm *, int *, ...);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiParallelSuperMng::
MpiParallelSuperMng(const ServiceBuildInfo& sbi)
: m_application(sbi.application())
, m_thread_mng(nullptr)
, m_stat(nullptr)
, m_is_parallel(false)
, m_rank(0)
, m_nb_rank(0)
, m_nb_local_sub_domain(1)
, m_mpi_main_communicator(MPI_COMM_NULL)
, m_main_communicator(MPI_COMM_NULL)
, m_adapter(nullptr)
, m_datatype_list(nullptr)
{
  m_thread_mng = new NullThreadMng();
  m_stat = Parallel::createDefaultStat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiParallelSuperMng::
~MpiParallelSuperMng()
{
  _finalize();

  try{
    delete m_datatype_list;
    if (m_adapter)
      m_adapter->destroy();
    delete m_stat;
    delete m_thread_mng;
  }
  catch(const Exception& ex){
    m_application->traceMng()->error() << ex;
  }

  MPI_Barrier(m_mpi_main_communicator);
  m_error_handler.removeHandler();

  MPI_Comm_free(&m_mpi_main_communicator);
  m_mpi_main_communicator = MPI_COMM_NULL;

  arcaneFinalizeMPI();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelSuperMng::
initialize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelSuperMng::
initMPI(IApplication* app)
{
  int* argc = nullptr;
  char*** argv = nullptr;

  Request::setNullRequest(Request(0,nullptr,MPI_REQUEST_NULL));
  Communicator::setNullCommunicator(Communicator(MPI_COMM_NULL));

  const CommandLineArguments& app_args = app->applicationInfo().commandLineArguments();
  argc = app_args.commandLineArgc();
  argv = app_args.commandLineArgv();

  // TODO:
  // Pouvoir utiliser un autre communicateur que MPI_COMM_WORLD
  int thread_wanted = MPI_THREAD_SERIALIZED;
  int thread_provided = 0;
  arcaneInitializeMPI(argc,argv,thread_wanted);

#ifndef ARCANE_USE_MPC
  // MPC (v 2.4.1) ne connait pas cette fonction
  MPI_Query_thread(&thread_provided);
#else
  thread_provided = MPI_THREAD_MULTIPLE;
#endif

  if (thread_provided < thread_wanted) {
    int my_rank = 0;
    // Affiche un message mais seulement un seul processeur.
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    if (my_rank==0)
      app->traceMng()->info() << "WARNING: MPI thread level provided!=wanted ("
                              << thread_provided << "!=" << thread_wanted << ")";
  } 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelSuperMng::
_ErrorHandler(MPI_Comm* comm, int* error_code, ...)
{
  ARCANE_UNUSED(comm);

  char error_buf[MPI_MAX_ERROR_STRING+1];
  int error_len = 0;
  int e = *error_code;
  // int MPI_Error_string(int errorcode, char *string, int *resultlen);
  MPI_Error_string(e,error_buf,&error_len);
  error_buf[error_len] = '\0';
  error_buf[MPI_MAX_ERROR_STRING] = '\0';
  
  // int MPI_Error_class(int errorcode, int *errorclass);

  ARCANE_FATAL("Error in MPI call code={0} msg={1}",*error_code,error_buf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelSuperMng::
build()
{
  ITraceMng* tm = m_application->traceMng();

  // TODO: Regarder s'il faut faire une réduction sur tous les temps.
  Real start_time = platform::getRealTime();
  initMPI(m_application);
  Real end_time = platform::getRealTime();

  MPI_Comm_dup(MPI_COMM_WORLD,&m_mpi_main_communicator);
  m_main_communicator = MP::Communicator(m_mpi_main_communicator);
  int rank, size;
  MPI_Comm_rank(m_mpi_main_communicator,&rank);
  MPI_Comm_size(m_mpi_main_communicator,&size);

#ifndef ARCANE_USE_MPC
  m_error_handler.registerHandler(m_main_communicator);
#endif

  if (rank==0){
    tm->info() << "MPI has non blocking collective";
    tm->info() << "MPI: sizeof(MPI_Count)=" << sizeof(MPI_Count);
    tm->info() << "MPI: is GPU Aware?=" << arcaneIsAcceleratorAwareMPI();
    tm->info() << "MPI: init_time (seconds)=" << (end_time-start_time);
  }

  m_rank = rank;
  m_nb_rank = size;
  m_is_parallel  = true;
  auto astat = m_stat->toArccoreStat();
  m_datatype_list = new MpiDatatypeList(false);
  m_adapter = new MpiAdapter(tm,astat,m_main_communicator,nullptr);
  auto c = createBuiltInDispatcher<Byte>(tm,nullptr,m_adapter,m_datatype_list);
  auto i32 = createBuiltInDispatcher<Int32>(tm,nullptr,m_adapter,m_datatype_list);
  auto i64 = createBuiltInDispatcher<Int64>(tm,nullptr,m_adapter,m_datatype_list);
  auto r = createBuiltInDispatcher<Real>(tm,nullptr,m_adapter,m_datatype_list);
  _setDispatchers(c,i32,i64,r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> MpiParallelSuperMng::
internalCreateWorldParallelMng(Int32 local_rank)
{
  ITraceMng* tm = m_application->traceMng();
  tm->debug()<<"[MpiParallelSuperMng::internalCreateWorldParallelMng]";
  if (local_rank!=0)
    ARCANE_THROW(ArgumentException,"local_rank has to be '0'");

  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_dup(m_main_communicator,&comm);

  int rank = -1;
  int nb_rank = -1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&nb_rank);

  bool is_parallel = nb_rank > 1;

  MpiParallelMngBuildInfo bi(comm);
  bi.is_parallel = is_parallel;
  bi.stat = m_stat;
  bi.trace_mng = tm;
  bi.timer_mng = nullptr;
  bi.thread_mng = m_thread_mng;
  bi.mpi_lock = nullptr;

  tm->debug()<<"[MpiParallelSuperMng::internalCreateWorldParallelMng] pm->build()";
  Ref<IParallelMng> pm = createRef<MpiParallelMng>(bi);
  pm->build();
  return pm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelSuperMng::
tryAbort()
{
  m_application->traceMng()->info() << "MpiParallelSuperMng: rank " << m_rank << " calling MPI_Abort";
  m_application->traceMng()->flush();
  MPI_Abort(m_main_communicator,2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelSuperMng::
barrier()
{
  MPI_Barrier(m_main_communicator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Superviseur en mode MPI avec un seul process alloué pour
 * se comporter comme en séqentiel mais avec un communicateur MPI qui
 * existe car on est compilé avec MPI.
 */
class MpiSequentialParallelSuperMng
: public SequentialParallelSuperMng
{
 public:
  explicit MpiSequentialParallelSuperMng(const ServiceBuildInfo& sbi)
  : SequentialParallelSuperMng(sbi,Parallel::Communicator(MPI_COMM_WORLD))
  {
  }

  ~MpiSequentialParallelSuperMng() override
  {
    arcaneFinalizeMPI();
  }

  void build() override
  {
    MpiParallelSuperMng::initMPI(application());
    SequentialParallelSuperMng::build();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MpiParallelSuperMng,
                        ServiceProperty("MpiParallelSuperMng",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelSuperMng));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MpiSequentialParallelSuperMng,
                        ServiceProperty("MpiSequentialParallelSuperMng",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelSuperMng));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
