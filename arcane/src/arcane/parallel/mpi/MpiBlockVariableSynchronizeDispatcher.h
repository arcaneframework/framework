// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiBlockVariableSynchronizeDispatcher.h                     (C) 2000-2022 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIBLOCKVARIABLESYNCHRONIZEDISPATCHER_H
#define ARCANE_PARALLEL_MPI_MPIBLOCKVARIABLESYNCHRONIZEDISPATCHER_H
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

class MpiBlockVariableSynchronizeDispatcherBuildInfo
{
 public:
  MpiBlockVariableSynchronizeDispatcherBuildInfo(MpiParallelMng* pm, GroupIndexTable* table,
                                                 Int32 block_size,Int32 nb_sequence)
  : m_parallel_mng(pm), m_table(table), m_block_size(block_size), m_nb_sequence(nb_sequence) { }
 public:
  MpiParallelMng* parallelMng() const{ return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
  //! Taille d'un bloc en octet.
  Int32 blockSize() const { return m_block_size; }
  //! Nombre de séquence de receive/send/wait
  Int32 nbSequence() const { return m_nb_sequence; }
 private:
  MpiParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
  Int32 m_block_size;
  Int32 m_nb_sequence;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation par block pour MPI de la synchronisation.
 *
 * Les messages sont envoyés par bloc d'une taille fixe.
 *
 * NOTE: cette optimisation respecte la norme MPI qui dit qu'on ne doit
 * plus toucher à la zone mémoire d'un message tant que celui-ci n'est
 * pas fini.
 */
template<typename SimpleType>
class MpiBlockVariableSynchronizeDispatcher
: public VariableSynchronizeDispatcher<SimpleType>
{
 public:

  using SyncBuffer = typename VariableSynchronizeDispatcher<SimpleType>::SyncBuffer;

 public:

  explicit MpiBlockVariableSynchronizeDispatcher(MpiBlockVariableSynchronizeDispatcherBuildInfo& bi);

 protected:

  void _beginSynchronize(SyncBuffer& sync_buffer) override;
  void _endSynchronize(SyncBuffer& sync_buffer) override;

 private:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
  Ref<Parallel::IRequestList> m_request_list;
  Int32 m_block_size;
  Int32 m_nb_sequence;

 private:

  bool _isSkipRank(Int32 rank,Int32 sequence) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
