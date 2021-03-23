// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryParallelSuperMng.h                              (C) 2000-2020 */
/*                                                                           */
/* Implémentation de 'IParallelSuperMng' mode mémoire partagé.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_SHAREDMEMORYPARALLELSUPERMNG_H
#define ARCANE_PARALLEL_THREAD_SHAREDMEMORYPARALLELSUPERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IParallelSuperMng.h"
#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ServiceBuildInfo;
}

namespace Arcane::MessagePassing
{

class SharedMemoryParallelMngContainer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Superviseur du parallélisme utilisant les threads
 */
class ARCANE_THREAD_EXPORT SharedMemoryParallelSuperMng
: public IParallelSuperMng
{
 public:

  explicit SharedMemoryParallelSuperMng(const ServiceBuildInfo& sbi);
  SharedMemoryParallelSuperMng(const ServiceBuildInfo& sbi,Parallel::Communicator communicator,
                               bool has_mpi_init);
  ~SharedMemoryParallelSuperMng() override;

  void initialize() override;
  void build() override;

  IApplication* application() const override { return m_application; }
  IThreadMng* threadMng() const override;
  bool isParallel() const override { return m_is_parallel; }
  Int32 commRank() const override { return 0; }
  Int32 commSize() const override { return 0; }
  Int32 traceRank() const override { return 0; }
  void* getMPICommunicator() override { return m_communicator.communicatorAddress(); }
  Parallel::Communicator communicator() const override { return m_communicator; }
  Ref<IParallelMng> internalCreateWorldParallelMng(Int32 local_rank) override;
  void tryAbort() override;
  bool isMasterIO() const override { return commRank()==0; }
  Int32 masterIORank() const override { return 0; }
  Int32 nbLocalSubDomain() override;
  void barrier() override {}

 public:

  void broadcast(ByteArrayView send_buf,Int32 rank) override;
  void broadcast(Int32ArrayView send_buf,Int32 rank) override;
  void broadcast(Int64ArrayView send_buf,Int32 rank) override;
  void broadcast(RealArrayView send_buf,Int32 rank) override;

 public:

  IApplication* m_application; //!< Gestionnaire principal
  Parallel::IStat* m_stat; //! Statistiques
  bool m_is_parallel;  //!< \a true si on est en mode parallèle
  SharedMemoryParallelMngContainer* m_container = nullptr;
  Ref<IParallelMngContainerFactory> m_builder_factory;
  Ref<IParallelMngContainer> m_main_builder;
  Parallel::Communicator m_communicator;
  bool m_has_mpi_init = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
