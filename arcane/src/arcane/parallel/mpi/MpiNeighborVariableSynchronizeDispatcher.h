// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiNeighborVariableSynchronizeDispatcher.h                  (C) 2000-2022 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
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
/*!
 * \brief Implémentation par neighbor pour MPI de la synchronisation.
 *
 * Les messages sont envoyés par bloc d'une taille fixe.
 *
 * NOTE: cette optimisation respecte la norme MPI qui dit qu'on ne doit
 * plus toucher à la zone mémoire d'un message tant que celui-ci n'est
 * pas fini.
 */
template <typename SimpleType>
class MpiNeighborVariableSynchronizeDispatcher
: public VariableSynchronizeDispatcher<SimpleType>
{
 public:

  using SyncBuffer = typename VariableSynchronizeDispatcher<SimpleType>::SyncBuffer;

 public:

  explicit MpiNeighborVariableSynchronizeDispatcher(MpiNeighborVariableSynchronizeDispatcherBuildInfo& bi);

  void compute() override;

 protected:

  void _beginSynchronize(SyncBuffer& sync_buffer) override;
  void _endSynchronize(SyncBuffer& sync_buffer) override;

 private:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
  UniqueArray<int> m_mpi_send_counts;
  UniqueArray<int> m_mpi_receive_counts;
  UniqueArray<int> m_mpi_send_displacements;
  UniqueArray<int> m_mpi_receive_displacements;
  MPI_Comm m_neighbor_communicator;
  Ref<IVariableSynchronizerMpiCommunicator> m_synchronizer_communicator;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
