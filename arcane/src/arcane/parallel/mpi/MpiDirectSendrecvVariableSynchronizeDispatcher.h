// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDirectSendrecvVariableSynchronizeDispatcher.h            (C) 2000-2023 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIDIRECTSENDRECVVARIABLESYNCHRONIZEDISPATCHER_H
#define ARCANE_PARALLEL_MPI_MPIDIRECTSENDRECVVARIABLESYNCHRONIZEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/ParallelMngDispatcher.h"

#include "arcane/parallel/mpi/ArcaneMpi.h"

#include "arcane/impl/VariableSynchronizer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiDirectSendrecvVariableSynchronizeDispatcherBuildInfo
{
 public:
  MpiDirectSendrecvVariableSynchronizeDispatcherBuildInfo(MpiParallelMng* pm, GroupIndexTable* table)
  : m_parallel_mng(pm), m_table(table) { }
 public:
  MpiParallelMng* parallelMng() const{ return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
 private:
  MpiParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de la synchronisation via MPI_Sendrecv.
 */
template<typename SimpleType>
class MpiDirectSendrecvVariableSynchronizeDispatcher
: public VariableSynchronizeDispatcher<SimpleType>
{
 public:

  using SyncBufferBase = VariableSynchronizeDispatcherSyncBufferBase;

 public:

  explicit MpiDirectSendrecvVariableSynchronizeDispatcher(MpiDirectSendrecvVariableSynchronizeDispatcherBuildInfo& bi);

 protected:

  void _beginSynchronize(SyncBufferBase& sync_buffer) override;
  void _endSynchronize(SyncBufferBase& sync_buffer) override;

 private:

  MpiParallelMng* m_mpi_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

