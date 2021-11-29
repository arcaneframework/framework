// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiVariableSynchronizeDispatcher.h                          (C) 2000-2021 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIVARIABLESYNCHRONIZEDISPATCHER_H
#define ARCANE_PARALLEL_MPI_MPIVARIABLESYNCHRONIZEDISPATCHER_H
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

class MpiVariableSynchronizeDispatcherBuildInfo
{
 public:
  MpiVariableSynchronizeDispatcherBuildInfo(MpiParallelMng* pm, GroupIndexTable* table)
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
 * \brief Implémentation optimisée pour MPI de la synchronisation.
 *
 * Par rapport à la version de base, cette implémentation fait un MPI_Waitsome
 * (au lieu d'un Waitall) et recopie dans le buffer de destination
 * dès qu'un message arrive.
 *
 * TODO: tester un type dérivé indexé pour l'entrée et la sortie
 * afin de ne pas faire la copie.
 * NOTE: cette optimisation respecte la norme MPI qui dit qu'on ne doit
 * plus toucher à la zone mémoire d'un message tant que celui-ci n'est
 * pas fini. L'optimisation proposé dans le TODO ne respecte plus
 * cette norme si on est en non bloquant, ou alors c'est au développeur
 * de module de vérifier que la variable en cours de synchronisation
 * n'est pas lue pendant la synchro non bloquante.
 */
template<typename SimpleType>
class MpiVariableSynchronizeDispatcher
: public VariableSynchronizeDispatcher<SimpleType>
{
 public:
  typedef typename VariableSynchronizeDispatcher<SimpleType>::SyncBuffer SyncBuffer;
 public:

  MpiVariableSynchronizeDispatcher(MpiVariableSynchronizeDispatcherBuildInfo& bi);

  ~MpiVariableSynchronizeDispatcher()
  {
    _destroyTypes();
  }

  void _destroyTypes()
  {
    Integer nb_share_type = m_share_derived_types.size();
    for( Integer i=0; i<nb_share_type; ++i ){
      MPI_Type_free(&m_share_derived_types[i]);
    }
    m_share_derived_types.clear();

    Integer nb_ghost_type = m_ghost_derived_types.size();
    for( Integer i=0; i<nb_ghost_type; ++i ){
      MPI_Type_free(&m_ghost_derived_types[i]);
    }
    m_ghost_derived_types.clear();
  }

  virtual void compute(ConstArrayView<VariableSyncInfo> sync_list);
  virtual void beginSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer);
  virtual void endSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer);

 private:
  MpiParallelMng* m_mpi_parallel_mng;
  UniqueArray<MPI_Request> m_send_requests;
  UniqueArray<MPI_Request> m_recv_requests;
  UniqueArray<Integer> m_recv_requests_done;
  UniqueArray<MPI_Datatype> m_share_derived_types;
  UniqueArray<MPI_Datatype> m_ghost_derived_types;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

