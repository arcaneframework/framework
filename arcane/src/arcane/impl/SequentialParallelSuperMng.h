// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SequentialParallelSuperMng.h                                (C) 2000-2020 */
/*                                                                           */
/* Superviseur du parallélisme en séquentiel.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_SEQUENTIALPARALLELSUPERMNGR_H
#define ARCANE_IMPL_SEQUENTIALPARALLELSUPERMNGR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/AbstractService.h"
#include "arcane/ParallelSuperMngDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Superviseur du parallélisme en mode séquentiel.
 *
 Dans ce mode, le parallélisme n'est pas supporté. Ce gestionnaire ne
 fait donc rien.
*/
class ARCANE_IMPL_EXPORT SequentialParallelSuperMng
: public AbstractService
, public ParallelSuperMngDispatcher
{
 public:

  // Construit un superviseur séquentiel lié au superviseur \a sm
  explicit SequentialParallelSuperMng(const ServiceBuildInfo& sbi);
  SequentialParallelSuperMng(const ServiceBuildInfo& sbi,Parallel::Communicator comm);
  ~SequentialParallelSuperMng() override;

  void build() override;
  void initialize() override;
  IApplication* application() const override { return m_application; }
  IThreadMng* threadMng() const override { return m_thread_mng; }
  bool isParallel() const override { return false; }
  Int32 commRank() const override { return 0; }
  Int32 commSize() const override { return 0; }
  Int32 traceRank() const override { return 0; }
  void* getMPICommunicator() override { return m_communicator.communicatorAddress(); }
  MP::Communicator communicator() const override { return m_communicator; }
  bool isMasterIO() const override { return true; }
  Integer masterIORank() const override { return 0; }
  Integer nbLocalSubDomain() override { return 1; }
  void barrier() override { }

  Ref<IParallelMng> internalCreateWorldParallelMng(Int32 local_rank) override;
  void tryAbort() override;

 private:
  
  IApplication* m_application; //!< Superviseur associé
  IThreadMng* m_thread_mng;
  ITimerMng* m_timer_mng;
  ScopedPtrT<ITimerMng> m_owned_timer_mng;
  MP::Communicator m_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

