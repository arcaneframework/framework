// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMpi.cc                                                (C) 2000-2026 */
/*                                                                           */
/* Global declarations for the MPI part of Arcane.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/ArcaneMpi.h"

#include "arcane/utils/String.h"
#include "arcane/impl/ArcaneMain.h"
#include "arcane/ApplicationBuildInfo.h"

#include <iostream>

// This file is used by OpenMpi to define extensions
// The page https://www.open-mpi.org/faq/?category=runcuda
// indicates how to detect if we are CUDA-AWARE.

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MPI_EXPORT bool
arcaneIsCudaAwareMPI()
{
  bool is_aware = false;
  // OpenMPI defines MPIX_CUDA_AWARE_SUPPORT and mpich defines MPIX_GPU_SUPPORT_CUDA
  // to indicate that MPIX_Query_cuda_support() is available.
#if defined(ARCANE_OS_LINUX)
#if defined(MPIX_CUDA_AWARE_SUPPORT) || defined(MPIX_GPU_SUPPORT_CUDA)
  is_aware =  (MPIX_Query_cuda_support()==1);
#endif
#endif
  return is_aware;
}

extern "C++" ARCANE_MPI_EXPORT bool
arcaneIsHipAwareMPI()
{
  bool is_aware = false;
  // OpenMPI defines MPIX_HIP_AWARE_SUPPORT and mpich defines MPIX_GPU_SUPPORT_HIP
  // to indicate that MPIX_Query_hip_support() is available.

  // MPICH
#if defined(ARCANE_OS_LINUX)
#if defined(MPIX_GPU_SUPPORT_HIP)
  // CRAY MPICH
#  if defined(CRAY_MPICH_VERSION)
  int is_supported = 0;
  MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP,&is_supported);
  is_aware = (is_supported!=0);
#  else
  is_aware =  (MPIX_Query_hip_support()==1);
#  endif
#endif

  // OpenMPI:
#if defined(MPIX_ROCM_AWARE_SUPPORT)
  is_aware =  (MPIX_Query_rocm_support()==1);
#endif
#endif
  return is_aware;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MPI_EXPORT bool
arcaneIsAcceleratorAwareMPI()
{
  return arcaneIsCudaAwareMPI() || arcaneIsHipAwareMPI();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Singleton class to automatically call MPI_Init and
 * MPI_Finalize if necessary.
 * MPI_Finalize is only called if we performed the init ourselves.
 */
class MpiAutoInit
{
 public:

  MpiAutoInit() = default;

 public:

  void initialize(int* argc,char*** argv,int wanted_thread_level)
  {
    int is_init = 0;
    MPI_Initialized(&is_init);

    if (is_init!=0)
      return;

    int thread_provided = 0;
    MPI_Init_thread(argc, argv, wanted_thread_level, &thread_provided);
    m_need_finalize = true;
  }

  void finalize()
  {
    if (m_need_finalize){
      MPI_Finalize();
      m_need_finalize = false;
    }
  }

 private:

  bool m_need_finalize = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AutoDetecterMPI
: public IApplicationBuildInfoVisitor
{
 public:
  void visit(ApplicationBuildInfo& app_build_info) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fills default values for message passing services.
 *
 * To know which service to use for message passing, it is
 * necessary to know the number of available MPI processes.
 * Unfortunately, this is not possible without initializing MPI, and
 * MPI initialization depends on the desired multi-threading level. To know this,
 * we must know if we want shared memory subdomains. If so, we try
 * init with MPI_THREAD_MULTIPLE. Otherwise, we use MPI_THREAD_SERIALIZED.
 * The parallelism manager will be responsible for checking if the available thread
 * level is sufficient (via MPI_Query_thread).
 */
void AutoDetecterMPI::
visit(ApplicationBuildInfo& app_build_info)
{
  String message_passing_service = app_build_info.messagePassingService();
  bool need_init = message_passing_service != "SequentialParallelSuperMng";
  bool has_shared_memory_message_passing = app_build_info.nbSharedMemorySubDomain()>0;

  // If MPI has not been initialized, we do it here.
  // We choose the thread level based on the number of
  // shared memory subdomains specified. If there is no memory
  // shared, it takes MPI_THREAD_SERIALIZED.
  int thread_wanted = MPI_THREAD_SERIALIZED;
  if (has_shared_memory_message_passing)
    thread_wanted = MPI_THREAD_MULTIPLE;

  int comm_size = 0;

  // We do not initialize if the requested service is 'Sequential'.
  if (need_init){
    // TODO: use the correct arguments.
    int* argc = nullptr;
    char*** argv = nullptr;
    arcaneInitializeMPI(argc,argv,thread_wanted);

    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  }

  // Sets the default exchange service.
  // Note that this will not be used if the user
  // has specified a service themselves
  message_passing_service = "Sequential";
  if (comm_size>1){
    if (has_shared_memory_message_passing)
      message_passing_service = "Hybrid";
    else
      message_passing_service = "Mpi";
  }
  else{
    if (has_shared_memory_message_passing)
      message_passing_service = "MpiSharedMemory";
    else
      message_passing_service = "MpiSequential";
  }
  message_passing_service = message_passing_service + "ParallelSuperMng";
  // Changes the default service.
  app_build_info.internalSetDefaultMessagePassingService(message_passing_service);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
MpiAutoInit global_mpi_auto_init;
AutoDetecterMPI global_autodetecter_mpi;
bool global_already_added = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCANE_MPI_EXPORT void
arcaneAutoDetectMessagePassingServiceMPI()
{
  if (!global_already_added)
    ArcaneMain::addApplicationBuildInfoVisitor(&global_autodetecter_mpi);
  global_already_added = true;
}

extern "C++" ARCANE_MPI_EXPORT void
arcaneInitializeMPI(int* argc,char*** argv,int wanted_thread_level)
{
  global_mpi_auto_init.initialize(argc,argv,wanted_thread_level);
}

extern "C++" ARCANE_MPI_EXPORT void
arcaneFinalizeMPI()
{
  global_mpi_auto_init.finalize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
