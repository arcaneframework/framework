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

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/VariableCollection.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IData.h"
#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/accelerator/core/Runner.h"

#include "arcane/impl/IBufferCopier.h"
#include "arcane/impl/IDataSynchronizeBuffer.h"

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de IDataSynchronizeBuffer pour une donnée
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

  Int64 totalReceiveSize() const final { return m_ghost_memory_view.bytes().size(); }
  Int64 totalSendSize() const final { return m_share_memory_view.bytes().size(); }

  void barrier() final { m_buffer_copier->barrier(); }

 public:

  void compute(IBufferCopier* copier, DataSynchronizeInfo* sync_list, Int32 datatype_size);
  IDataSynchronizeBuffer* genericBuffer() { return this; }

 protected:

  void _allocateBuffers(Int32 datatype_size);

 protected:

  DataSynchronizeInfo* m_sync_info = nullptr;
  //! Buffer pour toutes les données des entités fantômes qui serviront en réception
  MutableMemoryView m_ghost_memory_view;
  //! Buffer pour toutes les données des entités partagées qui serviront en envoi
  MutableMemoryView m_share_memory_view;

 protected:

  Int32 m_nb_rank = 0;
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
 * \brief Calcul et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 * \todo: ne pas allouer les tampons car leur conservation est couteuse en
 * terme de memoire.
 */
void VariableSynchronizeBufferBase::
compute(IBufferCopier* copier, DataSynchronizeInfo* sync_info, Int32 datatype_size)
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
/*!
 * \brief Implémentation de IDataSynchronizeBuffer pour une donnée
 */
class ARCANE_IMPL_EXPORT SingleDataSynchronizeBuffer
: public VariableSynchronizeBufferBase
{
 public:

  void copyReceiveAsync(Int32 index) final;
  void copySendAsync(Int32 index) final;

 public:

  void setDataView(MutableMemoryView v) { m_data_view = v; }
  MutableMemoryView dataView() { return m_data_view; }

 private:

  //! Vue sur les données de la variable
  MutableMemoryView m_data_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingleDataSynchronizeBuffer::
copyReceiveAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  MutableMemoryView var_values = dataView();
  ConstArrayView<Int32> indexes = m_sync_info->rankInfo(index).ghostIds();
  ConstMemoryView local_buffer = receiveBuffer(index);

  m_buffer_copier->copyFromBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingleDataSynchronizeBuffer::
copySendAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  ConstMemoryView var_values = dataView();
  Int32ConstArrayView indexes = m_sync_info->rankInfo(index).shareIds();
  MutableMemoryView local_buffer = sendBuffer(index);
  m_buffer_copier->copyToBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizerDispatcherBase
{
 public:

  explicit VariableSynchronizerDispatcherBase(const VariableSynchronizeDispatcherBuildInfo& bi);
  ~VariableSynchronizerDispatcherBase();

 protected:

  IParallelMng* m_parallel_mng = nullptr;
  IBufferCopier* m_buffer_copier = nullptr;
  Ref<DataSynchronizeInfo> m_sync_info;
  Ref<IDataSynchronizeImplementation> m_implementation_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerDispatcherBase::
VariableSynchronizerDispatcherBase(const VariableSynchronizeDispatcherBuildInfo& bi)
: m_parallel_mng(bi.parallelMng())
, m_sync_info(bi.synchronizeInfo())
{
  ARCANE_CHECK_POINTER(bi.factory().get());
  m_implementation_instance = bi.factory()->createInstance();
  m_implementation_instance->setDataSynchronizeInfo(m_sync_info.get());

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

VariableSynchronizerDispatcherBase::
~VariableSynchronizerDispatcherBase()
{
  delete m_buffer_copier;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de la synchronisation pour une donnée.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizerDispatcher
: private ReferenceCounterImpl
, public VariableSynchronizerDispatcherBase
, public IVariableSynchronizeDispatcher
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  //! Gère les buffers d'envoi et réception pour la synchronisation
  using SyncBuffer = SingleDataSynchronizeBuffer;

 public:

  explicit VariableSynchronizerDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi)
  : VariableSynchronizerDispatcherBase(bi)
  {
  }

 public:

  void compute() final;
  void beginSynchronize(INumericDataInternal* data) override;
  void endSynchronize() override;

 protected:

  void _beginSynchronize(VariableSynchronizeBufferBase& sync_buffer)
  {
    m_implementation_instance->beginSynchronize(sync_buffer.genericBuffer());
  }
  void _endSynchronize(VariableSynchronizeBufferBase& sync_buffer)
  {
    m_implementation_instance->endSynchronize(sync_buffer.genericBuffer());
  }

 private:

  SyncBuffer m_sync_buffer;
  bool m_is_in_sync = false;
  bool m_is_empty_sync = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerDispatcher::
beginSynchronize(INumericDataInternal* data)
{
  ARCANE_CHECK_POINTER(data);

  MutableMemoryView mem_view = data->memoryView();
  Int32 full_datatype_size = mem_view.datatypeSize();

  if (m_is_in_sync)
    ARCANE_FATAL("_beginSynchronize() has already been called");
  m_is_in_sync = true;

  m_is_empty_sync = (mem_view.bytes().size() == 0);
  if (m_is_empty_sync)
    return;
  m_sync_buffer.compute(m_buffer_copier, m_sync_info.get(), full_datatype_size);
  m_sync_buffer.setDataView(mem_view);
  _beginSynchronize(m_sync_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerDispatcher::
endSynchronize()
{
  if (!m_is_in_sync)
    ARCANE_FATAL("No pending synchronize(). You need to call beginSynchronize() before");
  if (!m_is_empty_sync)
    _endSynchronize(m_sync_buffer);
  m_is_in_sync = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables.
 */
void VariableSynchronizerDispatcher::
compute()
{
  if (!m_sync_info)
    ARCANE_FATAL("The instance is not initialized. You need to call setDataSynchronizeInfo() before");
  m_implementation_instance->compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IVariableSynchronizeDispatcher> IVariableSynchronizeDispatcher::
create(const VariableSynchronizeDispatcherBuildInfo& build_info)
{
  return makeRef<IVariableSynchronizeDispatcher>(new VariableSynchronizerDispatcher(build_info));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de IDataSynchronizeBuffer pour plusieurs données.
 */
class ARCANE_IMPL_EXPORT MultiDataSynchronizeBuffer
: public TraceAccessor
, public VariableSynchronizeBufferBase
{

 public:

  MultiDataSynchronizeBuffer(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

 public:

  void copyReceiveAsync(Int32 index) final;
  void copySendAsync(Int32 index) final;

 public:

  void setNbData(Int32 nb_data)
  {
    m_data_views.resize(nb_data);
  }
  void setDataView(Int32 index, MutableMemoryView v) { m_data_views[index] = v; }

 private:

  //! Vue sur les données de la variable
  SmallArray<MutableMemoryView> m_data_views;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiDataSynchronizeBuffer::
copyReceiveAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  Int64 data_offset = 0;
  Span<const std::byte> local_buffer_bytes = receiveBuffer(index).bytes();
  Int32ConstArrayView indexes = m_sync_info->rankInfo(index).ghostIds();
  const Int64 nb_element = indexes.size();
  for( MutableMemoryView var_values : m_data_views ){
    Int32 datatype_size = var_values.datatypeSize();
    Int64 current_size_in_bytes = nb_element * datatype_size;
    Span<const std::byte> sub_local_buffer_bytes = local_buffer_bytes.subSpan(data_offset,current_size_in_bytes);
    ConstMemoryView local_buffer = makeConstMemoryView(sub_local_buffer_bytes.data(),datatype_size,nb_element);
    if (current_size_in_bytes!=0)
      m_buffer_copier->copyFromBufferAsync(indexes, local_buffer, var_values);
    data_offset += current_size_in_bytes;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiDataSynchronizeBuffer::
copySendAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  Int64 data_offset = 0;
  Span<std::byte> local_buffer_bytes = sendBuffer(index).bytes();
  Int32ConstArrayView indexes = m_sync_info->rankInfo(index).shareIds();
  const Int64 nb_element = indexes.size();
  for( ConstMemoryView var_values : m_data_views ){
    Int32 datatype_size = var_values.datatypeSize();
    Int64 current_size_in_bytes = nb_element * datatype_size;
    Span<std::byte> sub_local_buffer_bytes = local_buffer_bytes.subSpan(data_offset,current_size_in_bytes);
    MutableMemoryView local_buffer = makeMutableMemoryView(sub_local_buffer_bytes.data(),datatype_size,nb_element);
    if (current_size_in_bytes!=0)
      m_buffer_copier->copyToBufferAsync(indexes, local_buffer, var_values);
    data_offset += current_size_in_bytes;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Synchronisation d'une liste de variables.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizerMultiDispatcher
: public IVariableSynchronizerMultiDispatcher
{
 public:

  explicit VariableSynchronizerMultiDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi)
  : m_parallel_mng(bi.parallelMng())
  , m_sync_info(bi.synchronizeInfo())
  {
  }

  void synchronize(VariableCollection vars) override;

 private:

  IParallelMng* m_parallel_mng = nullptr;
  Ref<DataSynchronizeInfo> m_sync_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Synchronisation d'une liste de variables.
 *
 * \brief Version 2 qui utilise directement des buffers au lieu
 * d'un ISerializer.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizerMultiDispatcherV2
: public VariableSynchronizerDispatcherBase
, public IVariableSynchronizerMultiDispatcher
{
 public:

  explicit VariableSynchronizerMultiDispatcherV2(const VariableSynchronizeDispatcherBuildInfo& bi)
  : VariableSynchronizerDispatcherBase(bi)
  {
  }

  void synchronize(VariableCollection vars);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMultiDispatcher::
synchronize(VariableCollection vars)
{
  Ref<IParallelExchanger> exchanger{ ParallelMngUtils::createExchangerRef(m_parallel_mng) };
  Integer nb_rank = m_sync_info->size();
  Int32UniqueArray recv_ranks(nb_rank);
  for (Integer i = 0; i < nb_rank; ++i) {
    Int32 rank = m_sync_info->rankInfo(i).targetRank();
    exchanger->addSender(rank);
    recv_ranks[i] = rank;
  }
  exchanger->initializeCommunicationsMessages(recv_ranks);
  for (Integer i = 0; i < nb_rank; ++i) {
    ISerializeMessage* msg = exchanger->messageToSend(i);
    ISerializer* sbuf = msg->serializer();
    Int32ConstArrayView share_ids = m_sync_info->rankInfo(i).shareIds();
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
    Int32ConstArrayView ghost_ids = m_sync_info->rankInfo(i).ghostIds();
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

void VariableSynchronizerMultiDispatcherV2::
synchronize(VariableCollection vars)
{
  ITraceMng* tm = m_parallel_mng->traceMng();
  MultiDataSynchronizeBuffer buffer(tm);

  const Int32 nb_var = vars.count();
  buffer.setNbData(nb_var);

  // Récupère les emplacements mémoire des données des variables et leur taille
  Int32 all_datatype_size = 0;
  {
    Int32 index = 0;
    for (VariableCollection::Enumerator ivar(vars); ++ivar;) {
      IVariable* var = *ivar;
      INumericDataInternal* numapi = var->data()->_commonInternal()->numericData();
      if (!numapi)
        ARCANE_FATAL("Variable '{0}' can not be synchronized because it is not a numeric data",var->name());
      MutableMemoryView mem_view = numapi->memoryView();
      all_datatype_size += mem_view.datatypeSize();
      buffer.setDataView(index,mem_view);
      ++index;
    }
  }

  buffer.compute(m_buffer_copier,m_sync_info.get(),all_datatype_size);

  m_implementation_instance->setDataSynchronizeInfo(m_sync_info.get());
  m_implementation_instance->compute();
  m_implementation_instance->beginSynchronize(buffer.genericBuffer());
  m_implementation_instance->endSynchronize(buffer.genericBuffer());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
: public AbstractDataSynchronizeImplementation
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
: public IDataSynchronizeImplementationFactory
{
 public:

  explicit Factory(IParallelMng* pm)
  : m_parallel_mng(pm)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new SimpleVariableSynchronizerDispatcher(this);
    return makeRef<IDataSynchronizeImplementation>(x);
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

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateSimpleVariableSynchronizerFactory(IParallelMng* pm)
{
  auto* x = new SimpleVariableSynchronizerDispatcher::Factory(pm);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleVariableSynchronizerDispatcher::
beginSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  IParallelMng* pm = m_parallel_mng;

  const bool use_blocking_send = false;
  auto* sync_info = _syncInfo();
  ARCANE_CHECK_POINTER(sync_info);
  Int32 nb_message = sync_info->size();

  /*pm->traceMng()->info() << " ** ** COMMON BEGIN SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/

  // Envoie les messages de réception non bloquant
  for (Integer i = 0; i < nb_message; ++i) {
    const VariableSyncInfo& vsi = sync_info->rankInfo(i);
    auto buf = _toLegacySmallView(vs_buf->receiveBuffer(i));
    if (!buf.empty()) {
      Parallel::Request rval = pm->recv(buf, vsi.targetRank(), false);
      m_all_requests.add(rval);
    }
  }

  vs_buf->copyAllSend();

  // Envoie les messages d'envoi en mode non bloquant.
  for (Integer i = 0; i < nb_message; ++i) {
    const VariableSyncInfo& vsi = sync_info->rankInfo(i);
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

IVariableSynchronizerMultiDispatcher* IVariableSynchronizerMultiDispatcher::
create(const VariableSynchronizeDispatcherBuildInfo& bi)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_LEGACY_MULTISYNCHRONIZE", true))
    if (v.value()>=1)
      return new VariableSynchronizerMultiDispatcher(bi);
  return new VariableSynchronizerMultiDispatcherV2(bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
