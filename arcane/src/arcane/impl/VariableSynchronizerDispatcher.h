// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerDispatcher.h                            (C) 2000-2023 */
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
#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/impl/IGenericVariableSynchronizerDispatcher.h"

#include "arcane/DataTypeDispatchingDataVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizer;
class VariableSynchronizerMultiDispatcher;
class Timer;

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

  VariableSyncInfo(Int32ConstArrayView share_ids, Int32ConstArrayView ghost_ids, Int32 rank);
  VariableSyncInfo(const VariableSyncInfo& rhs);
  VariableSyncInfo();

 public:

  //! Rang du processeur cible
  Int32 targetRank() const { return m_target_rank; }

  //! localIds() des entités à envoyer au rang targetRank()
  ConstArrayView<Int32> shareIds() const { return m_share_ids; }
  //! localIds() des entités à réceptionner du rang targetRank()
  ConstArrayView<Int32> ghostIds() const { return m_ghost_ids; }

  //! Nombre d'entités partagées
  Int32 nbShare() const { return m_share_ids.size(); }
  //! Nombre d'entités fantômes
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

  void _changeIds(Array<Int32>& ids, Int32ConstArrayView old_to_new_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations nécessaires pour synchroniser les entités sur un groupe.
 *
 * Il faut appeler recompute() après avoir ajouté ou modifier les instances
 * de VariableSyncInfo.
 */
class ARCANE_IMPL_EXPORT ItemGroupSynchronizeInfo
{
 public:

  ConstArrayView<VariableSyncInfo> infos() const { return m_ranks_info; }
  ArrayView<VariableSyncInfo> infos() { return m_ranks_info; }
  VariableSyncInfo& operator[](Int32 i) { return m_ranks_info[i]; }
  const VariableSyncInfo& operator[](Int32 i) const { return m_ranks_info[i]; }
  VariableSyncInfo& rankInfo(Int32 i) { return m_ranks_info[i]; }
  const VariableSyncInfo& rankInfo(Int32 i) const { return m_ranks_info[i]; }
  void clear() { m_ranks_info.clear(); }
  Int32 size() const { return m_ranks_info.size(); }
  void add(const VariableSyncInfo& s) { m_ranks_info.add(s); }
  void recompute();
  Int64 shareDisplacement(Int32 index) const { return m_share_displacements_base[index]; }
  Int64 ghostDisplacement(Int32 index) const { return m_ghost_displacements_base[index]; }
  Int64 totalNbGhost() const { return m_total_nb_ghost; }
  Int64 totalNbShare() const { return m_total_nb_share; }

 private:

  UniqueArray<VariableSyncInfo> m_ranks_info;
  //! Déplacement dans le buffer fantôme de chaque rang
  UniqueArray<Int64> m_ghost_displacements_base;
  //! Déplacement dans le buffer partagé de chaque rang
  UniqueArray<Int64> m_share_displacements_base;
  Int64 m_total_nb_ghost = 0;
  Int64 m_total_nb_share = 0;
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
  virtual void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info) =0;
  virtual void compute() =0;
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

  VariableSynchronizeDispatcherBuildInfo(IParallelMng* pm, GroupIndexTable* table,
                                         Ref<IGenericVariableSynchronizerDispatcherFactory> factory)
  : m_parallel_mng(pm)
  , m_table(table)
  , m_factory(factory)
  {}

 public:

  IParallelMng* parallelMng() const { return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
  Ref<IGenericVariableSynchronizerDispatcherFactory> factory() const
  {
    return m_factory;
  }

 private:

  IParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
  Ref<IGenericVariableSynchronizerDispatcherFactory> m_factory;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de IDataSynchronizeBuffer pour les variables
 */
class ARCANE_IMPL_EXPORT VariableSynchronizeBufferBase
: public IDataSynchronizeBuffer
{
 public:

  Int32 nbRank() const final { return m_nb_rank; }
  bool hasGlobalBuffer() const final { return true; }

  MutableMemoryView receiveBuffer(Int32 index) final { return _ghostLocalBuffer(index); }
  MutableMemoryView sendBuffer(Int32 index) final { return _shareLocalBuffer(index); }

  Int64 receiveDisplacement(Int32 index) const final { return _ghostDisplacementBase(index) * m_datatype_size; }
  Int64 sendDisplacement(Int32 index) const final { return _shareDisplacementBase(index) * m_datatype_size; }

  MutableMemoryView globalReceiveBuffer() final { return m_ghost_memory_view; }
  MutableMemoryView globalSendBuffer() final { return m_share_memory_view; }

  void copyReceiveAsync(Integer index) final;
  void copySendAsync(Integer index) final;
  Int64 totalReceiveSize() const final { return m_ghost_memory_view.bytes().size(); }
  Int64 totalSendSize() const final { return m_share_memory_view.bytes().size(); }

  void barrier() final { m_buffer_copier->barrier(); }

 public:

  void compute(IBufferCopier* copier, ItemGroupSynchronizeInfo* sync_list, Int32 datatype_size);
  IDataSynchronizeBuffer* genericBuffer() { return this; }
  void setDataView(MutableMemoryView v) { m_data_view = v; }
  MutableMemoryView dataMemoryView() { return m_data_view; }

 protected:

  void _allocateBuffers(Int32 datatype_size);

 protected:

  ItemGroupSynchronizeInfo* m_sync_info = nullptr;
  //! Buffer pour toutes les données des entités fantômes qui serviront en réception
  MutableMemoryView m_ghost_memory_view;
  //! Buffer pour toutes les données des entités partagées qui serviront en envoi
  MutableMemoryView m_share_memory_view;

 private:

  Int32 m_nb_rank = 0;
  //! Vue sur les données de la variable
  MutableMemoryView m_data_view;
  IBufferCopier* m_buffer_copier = nullptr;

  //! Buffer contenant les données concaténées en envoi et réception
  UniqueArray<std::byte> m_buffer;

  Int32 m_datatype_size = 0;

 private:

  Int64 _ghostDisplacementBase(Int32 index) const
  {
    return m_sync_info->ghostDisplacement(index);
  }
  Int64 _shareDisplacementBase(Int32 index) const
  {
    return m_sync_info->shareDisplacement(index);
  }

  Int32 _nbGhost(Int32 index) const
  {
    return m_sync_info->rankInfo(index).nbGhost();
  }

  Int32 _nbShare(Int32 index) const
  {
    return m_sync_info->rankInfo(index).nbShare();
  }

  MutableMemoryView _shareLocalBuffer(Int32 index) const
  {
    Int64 displacement = _shareDisplacementBase(index);
    Int32 local_size = _nbShare(index);
    return m_share_memory_view.subView(displacement, local_size);
  }
  MutableMemoryView _ghostLocalBuffer(Int32 index) const
  {
    Int64 displacement = _ghostDisplacementBase(index);
    Int32 local_size = _nbGhost(index);
    return m_ghost_memory_view.subView(displacement, local_size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de la synchronisation pour un type de donnée \a SimpleType.
 */
template <class SimpleType>
class ARCANE_IMPL_EXPORT VariableSynchronizeDispatcher
: public IDataTypeDataDispatcherT<SimpleType>
, public IVariableSynchronizeDispatcher
{
 public:

  //! Gère les buffers d'envoi et réception pour la synchronisation
  using SyncBuffer = VariableSynchronizeBufferBase;

 public:

  explicit VariableSynchronizeDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi);
  ~VariableSynchronizeDispatcher() override;

 public:

  void applyDispatch(IScalarDataT<SimpleType>* data) override;
  void applyDispatch(IArrayDataT<SimpleType>* data) override;
  void applyDispatch(IArray2DataT<SimpleType>* data) override;

 public:

  void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info) final;
  void compute() final;

 protected:

  void _beginSynchronize(VariableSynchronizeBufferBase& sync_buffer)
  {
    m_generic_instance->beginSynchronize(sync_buffer.genericBuffer());
  }
  void _endSynchronize(VariableSynchronizeBufferBase& sync_buffer)
  {
    m_generic_instance->endSynchronize(sync_buffer.genericBuffer());
  }

 private:

  IParallelMng* m_parallel_mng = nullptr;
  IBufferCopier* m_buffer_copier = nullptr;
  ItemGroupSynchronizeInfo* m_sync_info = nullptr;
  SyncBuffer m_1d_buffer;
  SyncBuffer m_2d_buffer;
  bool m_is_in_sync = false;
  Ref<IGenericVariableSynchronizerDispatcherFactory> m_factory;
  Ref<IGenericVariableSynchronizerDispatcher> m_generic_instance;

 private:

  void _applyDispatch(IData* data,SyncBuffer& sync_buffer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT VariableSynchronizerDispatcher
{
 public:
  typedef DataTypeDispatchingDataVisitor<IVariableSynchronizeDispatcher> DispatcherType;
 public:
  VariableSynchronizerDispatcher(IParallelMng* pm,DispatcherType* dispatcher)
  : m_parallel_mng(pm), m_dispatcher(dispatcher)
  {
  }
  ~VariableSynchronizerDispatcher();
  void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info);
  void compute();
  IDataVisitor* visitor() { return m_dispatcher; }
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
class ARCANE_IMPL_EXPORT VariableSynchronizerMultiDispatcher
{
 public:
  explicit VariableSynchronizerMultiDispatcher(IParallelMng* pm)
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
