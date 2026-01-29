// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMpi.cc                                                (C) 2000-2026 */
/*                                                                           */
/* Déclarations globales pour la partie MPI de Arcane.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/ArcaneMpi.h"

#include "arcane/utils/String.h"
#include "arcane/impl/ArcaneMain.h"
#include "arcane/ApplicationBuildInfo.h"

#include <iostream>

// Ce fichier est utilisé par OpenMpi pour définir des extensions
// La page https://www.open-mpi.org/faq/?category=runcuda
// indique comment détecter si on est CUDA-AWARE.

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
  // OpenMPI définit MPIX_CUDA_AWARE_SUPPORT et mpich définit MPIX_GPU_SUPPORT_CUDA
  // pour indiquer que MPIX_Query_cuda_support() est disponible.
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
  // OpenMPI définit MPIX_HIP_AWARE_SUPPORT et mpich définit MPIX_GPU_SUPPORT_HIP
  // pour indiquer que MPIX_Query_hip_support() est disponible.

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
 * \brief Classe singleton pour appeler automatiquement MPI_Init et
 * MPI_Finalize si besoin.
 * On appelle MPI_Finalize que si on a nous même fait l'init.
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
 * \brief Remplit les valeurs par défaut pour les services d'échange de message.
 *
 * Pour connaitre le service à utiliser pour l'échange de message, il est
 * nécessaire de connaitre le nombre de processus MPI disponible.
 * Malheureusement, cela n'est pas possible sans initialiser MPI, et
 * l'initialisation de MPI est dépendante du niveau de multi-threading
 * qu'on souhaite. Pour connaitre ce dernier, il faut savoir si on veut
 * avoir des sous-domaines en mémoire partagé. Si c'est le cas, alors on
 * essaie l'init avec MPI_THREAD_MULTIPLE. Sinon, on utilise MPI_THREAD_SERIALIZED.
 * Le gestionnaire de parallélisme ser chargera de vérifier si le niveau
 * de thread disponible est suffisant (via MPI_Query_thread).
 */
void AutoDetecterMPI::
visit(ApplicationBuildInfo& app_build_info)
{
  String message_passing_service = app_build_info.messagePassingService();
  bool need_init = message_passing_service != "SequentialParallelSuperMng";
  bool has_shared_memory_message_passing = app_build_info.nbSharedMemorySubDomain()>0;

  // Si MPI n'a pas été initialisé, on le fait ici.
  // On choisit le niveau de thread en fonction du nombre de
  // sous-domaines en mémoire partagée spécifié. Si pas de mémoire
  // partagée, prend MPI_THREAD_SERIALIZED.
  int thread_wanted = MPI_THREAD_SERIALIZED;
  if (has_shared_memory_message_passing)
    thread_wanted = MPI_THREAD_MULTIPLE;

  int comm_size = 0;

  // On ne fait pas l'initialisation si le service demandé est 'Sequential'.
  if (need_init){
    // TODO: utiliser les bons arguments.
    int* argc = nullptr;
    char*** argv = nullptr;
    arcaneInitializeMPI(argc,argv,thread_wanted);

    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  }

  // Positionne le service d'échange par défaut.
  // A noter que ce ne sera pas utilisé si l'utilisateur
  // a lui même spécifié un service
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
  // Change le service par défaut.
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
