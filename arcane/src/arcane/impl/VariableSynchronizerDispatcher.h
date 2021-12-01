// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerDispatcher.h                            (C) 2000-2021 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
#define ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Event.h"

#include "arcane/Parallel.h"
#include "arcane/ItemGroup.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/IParallelMng.h"

#include "arcane/impl/IBufferCopier.h"

#include "arcane/DataTypeDispatchingDataVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizer;
class Timer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class SimpleType>
class VariableSynchronize1D;
class VariableSynchronizerMultiDispatcher;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la liste des entités partagées/fantômes pour
 * un rang donné pour une synchronisation.
 *
 * TODO: Utiliser pour toutes les VariableSyncInfo un seul tableau pour les
 *       entités partagées et un seul tableau pour les entités fantômes qui
 *       sera géré par ItemGroupSynchronizeInfo.
 */
class ARCANE_IMPL_EXPORT VariableSyncInfo
{
 public:

  VariableSyncInfo() = default;
  VariableSyncInfo(Int32ConstArrayView share_ids,Int32ConstArrayView ghost_ids,
                   Int32 rank)
  : m_share_ids(share_ids), m_ghost_ids(ghost_ids),
    m_target_rank(rank) {}
	
 public:

  //! Rang du processeur cible
  Int32 targetRank() const { return m_target_rank; }

  //! localIds() des entités à envoyer au rang targetRank()
  ConstArrayView<Int32> shareIds() const { return m_share_ids; }
  //! localIds() des entités à réceptionner du rang targetRank()
  ConstArrayView<Int32> ghostIds() const { return m_ghost_ids; }

  Int32 nbShare() const { return m_share_ids.size(); }
  Int32 nbGhost() const { return m_ghost_ids.size(); }

  //! Met à jour les informations lorsque les localId() des entités changent
  void changeLocalIds(Int32ConstArrayView old_to_new_ids);

 private:

  //! localIds() des entités à envoyer au processeur #m_rank
  UniqueArray<Int32> m_share_ids;
  //! localIds() des entités à réceptionner du processeur #m_rank
  UniqueArray<Int32> m_ghost_ids;
  //! Rang du processeur cible
  Int32 m_target_rank = A_NULL_RANK;

 private:

  void _changeIds(Array<Int32>& ids,Int32ConstArrayView old_to_new_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations nécessaires pour synchroniser les entités sur un groupe.
 */
class ARCANE_IMPL_EXPORT ItemGroupSynchronizeInfo
{
 public:
  ConstArrayView<VariableSyncInfo> infos() const { return m_ranks_info; }
  ArrayView<VariableSyncInfo> infos() { return m_ranks_info; }
  VariableSyncInfo& operator[](Int32 i) { return m_ranks_info[i]; }
  const VariableSyncInfo& operator[](Int32 i) const { return m_ranks_info[i]; }
  void clear() { m_ranks_info.clear(); }
  Int32 size() const { return m_ranks_info.size(); }
  void add(const VariableSyncInfo& s) { m_ranks_info.add(s); }
 private:
  UniqueArray<VariableSyncInfo> m_ranks_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer l'envoi de la synchronisation.
 */
class ARCANE_IMPL_EXPORT IVariableSynchronizeDispatcher
{
 public:
  typedef FalseType HasStringDispatch;
 public:
  virtual ~IVariableSynchronizeDispatcher() = default;
 public:
  virtual void compute(ItemGroupSynchronizeInfo* sync_list) =0;
 protected:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un VariableSynchronizeDispatcher.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizeDispatcherBuildInfo
{
 public:
  VariableSynchronizeDispatcherBuildInfo(IParallelMng* pm, GroupIndexTable* table)
  : m_parallel_mng(pm), m_table(table) { }
 public:
  IParallelMng* parallelMng() const{ return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
 private:
  IParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de la synchronisation pour un type de donnée \a SimpleType.
 *
 * Cette classe est abstraite. La classe dérivée doit fournir une implémentation
 * de beginSynchronize() et endSynchronize().
 */
template<class SimpleType>
class ARCANE_IMPL_EXPORT VariableSynchronizeDispatcher
: public IDataTypeDataDispatcherT<SimpleType>
, public IVariableSynchronizeDispatcher
{
 public:
  //! Gère les buffers d'envoie et réception pour la synchronisation
  class SyncBuffer
  {
   public:
    void compute(IBufferCopier<SimpleType>* copier,ItemGroupSynchronizeInfo* sync_list,Int32 dim2_size);
   public:
    Int32 dim2Size() const { return m_dim2_size; }
    ArrayView<SimpleType> ghostBuffer(Int32 index) { return m_ghost_locals_buffer[index]; }
    ArrayView<SimpleType> shareBuffer(Int32 index) { return m_share_locals_buffer[index]; }
    ConstArrayView<SimpleType> ghostBuffer(Int32 index) const { return m_ghost_locals_buffer[index]; }
    ConstArrayView<SimpleType> shareBuffer(Int32 index) const { return m_share_locals_buffer[index]; }
    void setDataView(ArrayView<SimpleType> v) { m_data_view = v; }
    ArrayView<SimpleType> dataView() { return m_data_view; }
    void copyReceive(Integer index);
    void copySend(Integer index);
   private:
    Integer m_dim2_size = 0;
    //! Buffer pour toutes les données des entités fantômes qui serviront en réception
    UniqueArray<SimpleType> m_ghost_buffer;
    //! Buffer pour toutes les données des entités partagées qui serviront en envoi
    UniqueArray<SimpleType> m_share_buffer;
    //! Position dans \a m_ghost_buffer de chaque rang
    UniqueArray< ArrayView<SimpleType> > m_ghost_locals_buffer;
    //! Position dans \a m_ghost_buffer de chaque rang
    UniqueArray< ArrayView<SimpleType> > m_share_locals_buffer;
    ItemGroupSynchronizeInfo* m_sync_info = nullptr;
    //! Vue sur les données de la variable
    ArrayView<SimpleType> m_data_view;
    IBufferCopier<SimpleType>* m_buffer_copier = nullptr;
  };

 public:

  VariableSynchronizeDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi);
  ~VariableSynchronizeDispatcher() override;

 public:

  void applyDispatch(IScalarDataT<SimpleType>* data) override;
  void applyDispatch(IArrayDataT<SimpleType>* data) override;
  void applyDispatch(IArray2DataT<SimpleType>* data) override;
  void applyDispatch(IMultiArray2DataT<SimpleType>* data) override;
  void compute(ItemGroupSynchronizeInfo* sync_list) override;

 protected:

  virtual void _beginSynchronize(SyncBuffer& sync_buffer) =0;
  virtual void _endSynchronize(SyncBuffer& sync_buffer) =0;

 protected:

  IParallelMng* m_parallel_mng = nullptr;
  IBufferCopier<SimpleType>* m_buffer_copier = nullptr;
  ItemGroupSynchronizeInfo* m_sync_info = nullptr;
  //TODO: a supprimer car l'information est dans \a m_sync_info;
  ConstArrayView<VariableSyncInfo> m_sync_list;
  SyncBuffer m_1d_buffer;
  SyncBuffer m_2d_buffer;
  bool m_is_in_sync = false;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation basique de la sérialisation.
 *
 * Cette implémentation est faite à partir de send/receive suivi de 'wait'.
 */
template<class SimpleType>
class ARCANE_IMPL_EXPORT SimpleVariableSynchronizeDispatcher
: public VariableSynchronizeDispatcher<SimpleType>
{
  using BaseClass = VariableSynchronizeDispatcher<SimpleType>;
  using SyncBuffer = typename VariableSynchronizeDispatcher<SimpleType>::SyncBuffer;
  using BaseClass::m_parallel_mng;
 public:
  SimpleVariableSynchronizeDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi)
  : BaseClass(bi){}

 protected:

  void _beginSynchronize(SyncBuffer& sync_buffer) override;
  void _endSynchronize(SyncBuffer& sync_buffer) override;

 private:

  UniqueArray<Parallel::Request> m_all_requests;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizerDispatcher
{
 public:
  typedef DataTypeDispatchingDataVisitor<IVariableSynchronizeDispatcher> DispatcherType;
 public:
  VariableSynchronizerDispatcher(IParallelMng* pm,DispatcherType* dispatcher)
  : m_parallel_mng(pm), m_dispatcher(dispatcher)
  {
  }
  ~VariableSynchronizerDispatcher();
  void compute(ItemGroupSynchronizeInfo* sync_info);
  IDataVisitor* visitor()
  {
    return m_dispatcher;
  }
 private:
  IParallelMng* m_parallel_mng;
  DispatcherType* m_dispatcher;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Synchronisation d'une liste de variables.
 */
class VariableSynchronizerMultiDispatcher
{
 public:
  VariableSynchronizerMultiDispatcher(IParallelMng* pm)
  : m_parallel_mng(pm)
  {
  }

  void synchronize(VariableCollection vars,ConstArrayView<VariableSyncInfo> sync_infos);
 private:
  IParallelMng* m_parallel_mng;
};
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespcae Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
