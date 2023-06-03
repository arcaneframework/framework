// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerDispatcher.cc                           (C) 2000-2023 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/VariableSynchronizerDispatcher.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Array2View.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"

#include "arcane/core/VariableCollection.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/IData.h"
#include "arcane/core/datatype/DataStorageTypeInfo.h"
#include "arcane/core/datatype/DataTypeTraits.h"
#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/accelerator/core/Runner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  ArrayView<Byte>
  _toLegacySmallView(MutableMemoryView memory_view)
  {
    Span<std::byte> bytes = memory_view.bytes();
    void* data = bytes.data();
    Int32 size = bytes.smallView().size();
    return { size, reinterpret_cast<Byte*>(data) };
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: plutôt que d'utiliser la mémoire managée, il est préférable d'avoir
// une copie sur le device des IDs. Cela permettra d'éviter des transferts
// potentiels si on mélange synchronisation de variables sur accélérateurs et
// sur CPU.

VariableSyncInfo::
VariableSyncInfo()
: m_share_ids(platform::getDefaultDataAllocator())
, m_ghost_ids(platform::getDefaultDataAllocator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSyncInfo::
VariableSyncInfo(Int32ConstArrayView share_ids, Int32ConstArrayView ghost_ids,
                 Int32 rank)
: VariableSyncInfo()
{
  m_target_rank = rank;
  m_share_ids.copy(share_ids);
  m_ghost_ids.copy(ghost_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSyncInfo::
VariableSyncInfo(const VariableSyncInfo& rhs)
: VariableSyncInfo()
{
  // NOTE: pour l'instant (avril 2023) il faut un constructeur de recopie
  // explicite pour spécifier l'allocateur
  m_target_rank = rhs.m_target_rank;
  m_share_ids.copy(rhs.m_share_ids);
  m_ghost_ids.copy(rhs.m_ghost_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupSynchronizeInfo::
recompute()
{
  Integer nb_message = this->size();

  m_ghost_displacements_base.resize(nb_message);
  m_share_displacements_base.resize(nb_message);

  m_total_nb_ghost = 0;
  m_total_nb_share = 0;

  {
    Integer ghost_displacement = 0;
    Integer share_displacement = 0;
    Int32 index = 0;
    for (const VariableSyncInfo& vsi : infos()) {
      Int32 ghost_size = vsi.nbGhost();
      m_ghost_displacements_base[index] = ghost_displacement;
      ghost_displacement += ghost_size;
      Int32 share_size = vsi.nbShare();
      m_share_displacements_base[index] = share_displacement;
      share_displacement += share_size;
      ++index;
    }
    m_total_nb_ghost = ghost_displacement;
    m_total_nb_share = share_displacement;
  }
}


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
 * \brief Gestion de la synchronisation.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizerDispatcher
: public IVariableSynchronizeDispatcher
{
 public:

  //! Gère les buffers d'envoi et réception pour la synchronisation
  using SyncBuffer = VariableSynchronizeBufferBase;

 public:

  explicit VariableSynchronizerDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi);
  ~VariableSynchronizerDispatcher() override;

 public:

  void applyDispatch(IData* data) override;
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
  SyncBuffer m_sync_buffer;
  bool m_is_in_sync = false;
  Ref<IGenericVariableSynchronizerDispatcherFactory> m_factory;
  Ref<IGenericVariableSynchronizerDispatcher> m_generic_instance;

 private:

  void _applyDispatch(IData* data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerDispatcher::
VariableSynchronizerDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi)
: m_parallel_mng(bi.parallelMng())
, m_factory(bi.factory())
{
  ARCANE_CHECK_POINTER(m_factory.get());
  m_generic_instance = this->m_factory->createInstance();

  // TODO: Utiliser une unique instance de IBufferCopier partagée par tous les
  // dispatchers
  if (bi.table())
    m_buffer_copier = new TableBufferCopier(bi.table());
  else
    m_buffer_copier = new DirectBufferCopier();

  auto* internal_pm = m_parallel_mng->_internalApi();
  Runner* runner = internal_pm->defaultRunner();
  bool is_accelerator_aware = internal_pm->isAcceleratorAware();

  // Si le IParallelMng gère la mémoire des accélérateurs alors on alloue le
  // buffer sur le device. On pourrait utiliser le mémoire managée mais certaines
  // implémentations MPI (i.e: BXI) ne le supportent pas.
  if (runner && is_accelerator_aware) {
    m_buffer_copier->setRunQueue(internal_pm->defaultQueue());
    auto* a = platform::getDataMemoryRessourceMng()->getAllocator(eMemoryRessource::Device);
    m_buffer_copier->setAllocator(a);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerDispatcher::
~VariableSynchronizerDispatcher()
{
  delete m_buffer_copier;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerDispatcher::
_applyDispatch(IData* data)
{
  INumericDataInternal* numapi = data->_commonInternal()->numericData();
  if (!numapi)
    ARCANE_FATAL("Data can not be synchronized because it is not a numeric data");

  MutableMemoryView mem_view = numapi->memoryView();
  Int32 full_datatype_size = mem_view.datatypeSize();
  if (mem_view.bytes().size() == 0)
    return;

  if (m_is_in_sync)
    ARCANE_FATAL("Only one pending serialisation is supported");
  m_is_in_sync = true;
  m_sync_buffer.compute(m_buffer_copier, m_sync_info, full_datatype_size);
  m_sync_buffer.setDataView(mem_view);
  _beginSynchronize(m_sync_buffer);
  _endSynchronize(m_sync_buffer);
  m_is_in_sync = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerDispatcher::
applyDispatch(IData* data)
{
  _applyDispatch(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerDispatcher::
setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info)
{
  m_sync_info = sync_info;
  m_generic_instance->setItemGroupSynchronizeInfo(sync_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 */
void VariableSynchronizerDispatcher::
compute()
{
  if (!m_sync_info)
    ARCANE_FATAL("The instance is not initialized. You need to call setItemGroupSynchronizeInfo() before");
  m_generic_instance->compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizeDispatcher* IVariableSynchronizeDispatcher::
create(const VariableSynchronizeDispatcherBuildInfo& build_info)
{
  return new VariableSynchronizerDispatcher(build_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 * \todo: ne pas allouer les tampons car leur conservation est couteuse en
 * terme de memoire.
 */
void VariableSynchronizeBufferBase::
compute(IBufferCopier* copier, ItemGroupSynchronizeInfo* sync_info, Int32 datatype_size)
{
  m_datatype_size = datatype_size;
  m_buffer_copier = copier;
  m_sync_info = sync_info;
  m_nb_rank = sync_info->size();

  IMemoryAllocator* allocator = m_buffer_copier->allocator();
  if (allocator && allocator != m_buffer.allocator())
    m_buffer = UniqueArray<std::byte>(allocator);

  _allocateBuffers(datatype_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizeBufferBase::
copyReceiveAsync(Integer index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  MutableMemoryView var_values = dataMemoryView();
  const VariableSyncInfo& vsi = (*m_sync_info)[index];
  ConstArrayView<Int32> indexes = vsi.ghostIds();
  ConstMemoryView local_buffer = receiveBuffer(index);

  m_buffer_copier->copyFromBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizeBufferBase::
copySendAsync(Integer index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  ConstMemoryView var_values = dataMemoryView();
  const VariableSyncInfo& vsi = (*m_sync_info)[index];
  Int32ConstArrayView indexes = vsi.shareIds();
  MutableMemoryView local_buffer = sendBuffer(index);
  m_buffer_copier->copyToBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaires aux envois et réceptions
 * pour les synchronisations des variables 1D.
 *
 * \todo: ne pas converver les tampons pour chaque type de donnée des variables
 * car leur conservation est couteuse en terme de memoire.
 */
void VariableSynchronizeBufferBase::
_allocateBuffers(Int32 datatype_size)
{
  Int64 total_ghost_buffer = m_sync_info->totalNbGhost();
  Int64 total_share_buffer = m_sync_info->totalNbShare();

  Int32 full_dim2_size = datatype_size;
  m_buffer.resize((total_ghost_buffer + total_share_buffer) * full_dim2_size);

  Int64 share_offset = total_ghost_buffer * full_dim2_size;

  auto s1 = m_buffer.span().subspan(0, share_offset);
  m_ghost_memory_view = makeMutableMemoryView(s1.data(), full_dim2_size, total_ghost_buffer);
  auto s2 = m_buffer.span().subspan(share_offset, total_share_buffer * full_dim2_size);
  m_share_memory_view = makeMutableMemoryView(s2.data(), full_dim2_size, total_share_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMultiDispatcher::
synchronize(VariableCollection vars, ConstArrayView<VariableSyncInfo> sync_infos)
{
  Ref<IParallelExchanger> exchanger{ ParallelMngUtils::createExchangerRef(m_parallel_mng) };
  Integer nb_rank = sync_infos.size();
  Int32UniqueArray recv_ranks(nb_rank);
  for (Integer i = 0; i < nb_rank; ++i) {
    Int32 rank = sync_infos[i].targetRank();
    exchanger->addSender(rank);
    recv_ranks[i] = rank;
  }
  exchanger->initializeCommunicationsMessages(recv_ranks);
  for (Integer i = 0; i < nb_rank; ++i) {
    ISerializeMessage* msg = exchanger->messageToSend(i);
    ISerializer* sbuf = msg->serializer();
    Int32ConstArrayView share_ids = sync_infos[i].shareIds();
    sbuf->setMode(ISerializer::ModeReserve);
    for (VariableCollection::Enumerator ivar(vars); ++ivar;) {
      (*ivar)->serialize(sbuf, share_ids, nullptr);
    }
    sbuf->allocateBuffer();
    sbuf->setMode(ISerializer::ModePut);
    for (VariableCollection::Enumerator ivar(vars); ++ivar;) {
      (*ivar)->serialize(sbuf, share_ids, nullptr);
    }
  }
  exchanger->processExchange();
  for (Integer i = 0; i < nb_rank; ++i) {
    ISerializeMessage* msg = exchanger->messageToReceive(i);
    ISerializer* sbuf = msg->serializer();
    Int32ConstArrayView ghost_ids = sync_infos[i].ghostIds();
    sbuf->setMode(ISerializer::ModeGet);
    for (VariableCollection::Enumerator ivar(vars); ++ivar;) {
      (*ivar)->serialize(sbuf, ghost_ids, nullptr);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSyncInfo::
_changeIds(Array<Int32>& ids, Int32ConstArrayView old_to_new_ids)
{
  UniqueArray<Int32> orig_ids(ids);
  ids.clear();

  for (Integer z = 0, zs = orig_ids.size(); z < zs; ++z) {
    Int32 old_id = orig_ids[z];
    Int32 new_id = old_to_new_ids[old_id];
    if (new_id != NULL_ITEM_LOCAL_ID)
      ids.add(new_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSyncInfo::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  _changeIds(m_share_ids, old_to_new_ids);
  _changeIds(m_ghost_ids, old_to_new_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation basique de la sérialisation.
 *
 * Cette implémentation est faite à partir de send/receive suivi de 'wait'.
 */
class SimpleVariableSynchronizerDispatcher
: public AbstractGenericVariableSynchronizerDispatcher
{
 public:

  class Factory;
  explicit SimpleVariableSynchronizerDispatcher(Factory* f);

 protected:

  void compute() override {}
  void beginSynchronize(IDataSynchronizeBuffer* buf) override;
  void endSynchronize(IDataSynchronizeBuffer* buf) override;

 private:

  IParallelMng* m_parallel_mng = nullptr;
  UniqueArray<Parallel::Request> m_all_requests;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleVariableSynchronizerDispatcher::Factory
: public IGenericVariableSynchronizerDispatcherFactory
{
 public:

  explicit Factory(IParallelMng* pm)
  : m_parallel_mng(pm)
  {}

  Ref<IGenericVariableSynchronizerDispatcher> createInstance() override
  {
    auto* x = new SimpleVariableSynchronizerDispatcher(this);
    return makeRef<IGenericVariableSynchronizerDispatcher>(x);
  }

 public:

  IParallelMng* m_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleVariableSynchronizerDispatcher::
SimpleVariableSynchronizerDispatcher(Factory* f)
: m_parallel_mng(f->m_parallel_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IGenericVariableSynchronizerDispatcherFactory>
arcaneCreateSimpleVariableSynchronizerFactory(IParallelMng* pm)
{
  auto* x = new SimpleVariableSynchronizerDispatcher::Factory(pm);
  return makeRef<IGenericVariableSynchronizerDispatcherFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleVariableSynchronizerDispatcher::
beginSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  IParallelMng* pm = m_parallel_mng;

  const bool use_blocking_send = false;
  auto sync_list = _syncInfo()->infos();
  Int32 nb_message = sync_list.size();

  /*pm->traceMng()->info() << " ** ** COMMON BEGIN SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/

  // Envoie les messages de réception non bloquant
  for (Integer i = 0; i < nb_message; ++i) {
    const VariableSyncInfo& vsi = sync_list[i];
    auto buf = _toLegacySmallView(vs_buf->receiveBuffer(i));
    if (!buf.empty()) {
      Parallel::Request rval = pm->recv(buf, vsi.targetRank(), false);
      m_all_requests.add(rval);
    }
  }

  vs_buf->copyAllSend();

  // Envoie les messages d'envoi en mode non bloquant.
  for (Integer i = 0; i < nb_message; ++i) {
    const VariableSyncInfo& vsi = sync_list[i];
    auto buf = _toLegacySmallView(vs_buf->sendBuffer(i));

    //ConstArrayView<SimpleType> const_share = share_local_buffer;
    if (!buf.empty()) {
      //for( Integer i=0, is=share_local_buffer.size(); i<is; ++i )
      //trace->info() << "TO rank=" << vsi.m_target_rank << " I=" << i << " V=" << share_local_buffer[i]
      //                << " lid=" << share_grp[i] << " v2=" << var_values[share_grp[i]];
      Parallel::Request rval = pm->send(buf, vsi.targetRank(), use_blocking_send);
      if (!use_blocking_send)
        m_all_requests.add(rval);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleVariableSynchronizerDispatcher::
endSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  IParallelMng* pm = m_parallel_mng;

  /*pm->traceMng()->info() << " ** ** COMMON END SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/

  // Attend que les réceptions se terminent
  pm->waitAllRequests(m_all_requests);
  m_all_requests.clear();

  // Recopie dans la variable le message de retour.
  vs_buf->copyAllReceive();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDataSynchronizeBuffer::
copyAllSend()
{
  Int32 nb_rank = nbRank();
  for (Int32 i = 0; i < nb_rank; ++i)
    copySendAsync(i);
  barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDataSynchronizeBuffer::
copyAllReceive()
{
  Int32 nb_rank = nbRank();
  for (Int32 i = 0; i < nb_rank; ++i)
    copyReceiveAsync(i);
  barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DirectBufferCopier::
barrier()
{
  if (m_queue)
    m_queue->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
