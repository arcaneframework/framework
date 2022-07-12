// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiNeighborVariableSynchronizeDispatcher.h                  (C) 2000-2022 */
/*                                                                           */
/* Synchronisations des variables via MPI_Neighbor_alltoallv.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPINEIGHBORVARIABLESYNCHRONIZEDISPATCHER_H
#define ARCANE_PARALLEL_MPI_MPINEIGHBORVARIABLESYNCHRONIZEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/parallel/mpi/IVariableSynchronizerMpiCommunicator.h"

#include "arcane/impl/VariableSynchronizerDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiNeighborVariableSynchronizeDispatcherBuildInfo
{
 public:

  MpiNeighborVariableSynchronizeDispatcherBuildInfo(MpiParallelMng* pm, GroupIndexTable* table,
                                                    Ref<IVariableSynchronizerMpiCommunicator> synchronizer_communicator)
  : m_parallel_mng(pm)
  , m_table(table)
  , m_synchronizer_communicator(synchronizer_communicator)
  {}

 public:

  MpiParallelMng* parallelMng() const { return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
  Ref<IVariableSynchronizerMpiCommunicator> synchronizerCommunicator() const
  {
    return m_synchronizer_communicator;
  }

 private:

  MpiParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
  Ref<IVariableSynchronizerMpiCommunicator> m_synchronizer_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GenericMpiNeighborVariableSynchronizer
{
 public:
  explicit GenericMpiNeighborVariableSynchronizer(MpiNeighborVariableSynchronizeDispatcherBuildInfo& bi);
 public:
  void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info) { m_sync_info = sync_info; }
  void compute();
  void beginSynchronize(IVariableSynchronizerBuffer* buf);
  void endSynchronize(IVariableSynchronizerBuffer* buf);
 private:
  MpiParallelMng* m_mpi_parallel_mng = nullptr;
  UniqueArray<int> m_mpi_send_counts;
  UniqueArray<int> m_mpi_receive_counts;
  UniqueArray<int> m_mpi_send_displacements;
  UniqueArray<int> m_mpi_receive_displacements;
  Ref<IVariableSynchronizerMpiCommunicator> m_synchronizer_communicator;
  ItemGroupSynchronizeInfo* m_sync_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de la synchronisations des variables via
 * MPI_Neighbor_alltoallv().
 */
template <typename SimpleType>
class MpiNeighborVariableSynchronizeDispatcher
: public VariableSynchronizeDispatcher<SimpleType>
{
 public:

  using SyncBuffer = typename VariableSynchronizeDispatcher<SimpleType>::SyncBuffer;

 public:

  explicit MpiNeighborVariableSynchronizeDispatcher(MpiNeighborVariableSynchronizeDispatcherBuildInfo& bi);

  void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info) override
  {
    VariableSynchronizeDispatcher<SimpleType>::setItemGroupSynchronizeInfo(sync_info);
    m_generic.setItemGroupSynchronizeInfo(sync_info);
  }
  void compute() override;

 protected:

  void _beginSynchronize(SyncBuffer& sync_buffer) override;
  void _endSynchronize(SyncBuffer& sync_buffer) override;

 private:

  GenericMpiNeighborVariableSynchronizer m_generic;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
