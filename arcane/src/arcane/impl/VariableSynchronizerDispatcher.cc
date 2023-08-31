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
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"

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
#include "arcane/impl/DataSynchronizeInfo.h"
#include "arcane/impl/internal/DataSynchronizeBuffer.h"

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

class VariableSynchronizerDispatcherBase
{
 public:

  explicit VariableSynchronizerDispatcherBase(const VariableSynchronizeDispatcherBuildInfo& bi);
  ~VariableSynchronizerDispatcherBase();

 protected:

  IParallelMng* m_parallel_mng = nullptr;
  IBufferCopier* m_buffer_copier = nullptr;
  Runner* m_runner = nullptr;
  Ref<DataSynchronizeInfo> m_sync_info;
  Ref<IDataSynchronizeImplementation> m_implementation_instance;

 protected:

  void _setCurrentDevice();
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
    m_runner = runner;
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
/*!
 * \brief Positionne le device associé à notre RunQueue comme le device courant.
 *
 * Si on utilise une RunQueue, positionne le device associé à celui
 * de cette RunQueue. Cela permet de garantir que les allocations mémoires
 * effectuées lors des synchronisations seront sur le bon device.
 */
void VariableSynchronizerDispatcherBase::
_setCurrentDevice()
{
  if (m_runner)
    m_runner->setAsCurrentDevice();
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

  explicit VariableSynchronizerDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi)
  : VariableSynchronizerDispatcherBase(bi)
  , m_sync_buffer(m_sync_info.get(), m_buffer_copier)
  {
  }

 public:

  void compute() final;
  void beginSynchronize(INumericDataInternal* data, bool is_compare_sync) override;
  DataSynchronizeResult endSynchronize() override;

 protected:

  void _beginSynchronize(DataSynchronizeBufferBase& sync_buffer)
  {
    m_implementation_instance->beginSynchronize(&sync_buffer);
  }
  void _endSynchronize(DataSynchronizeBufferBase& sync_buffer)
  {
    m_implementation_instance->endSynchronize(&sync_buffer);
  }

 private:

  //! Gère les buffers d'envoi et réception pour la synchronisation
  SingleDataSynchronizeBuffer m_sync_buffer;
  bool m_is_in_sync = false;
  bool m_is_empty_sync = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerDispatcher::
beginSynchronize(INumericDataInternal* data, bool is_compare_sync)
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
  _setCurrentDevice();
  m_sync_buffer.setDataView(mem_view);
  m_sync_buffer.prepareSynchronize(full_datatype_size, is_compare_sync);
  _beginSynchronize(m_sync_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeResult VariableSynchronizerDispatcher::
endSynchronize()
{
  if (!m_is_in_sync)
    ARCANE_FATAL("No pending synchronize(). You need to call beginSynchronize() before");
  DataSynchronizeResult result;
  if (!m_is_empty_sync) {
    _setCurrentDevice();
    _endSynchronize(m_sync_buffer);
    result = m_sync_buffer.finalizeSynchronize();
  }
  m_is_in_sync = false;
  return result;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notifie l'implémentation que les informations de synchronisation
 * ont changé.
 */
void VariableSynchronizerDispatcher::
compute()
{
  if (!m_sync_info)
    ARCANE_FATAL("The instance is not initialized. You need to call setDataSynchronizeInfo() before");
  _setCurrentDevice();
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
    Int32 rank = m_sync_info->targetRank(i);
    exchanger->addSender(rank);
    recv_ranks[i] = rank;
  }
  exchanger->initializeCommunicationsMessages(recv_ranks);
  for (Integer i = 0; i < nb_rank; ++i) {
    ISerializeMessage* msg = exchanger->messageToSend(i);
    ISerializer* sbuf = msg->serializer();
    Int32ConstArrayView share_ids = m_sync_info->sendInfo().localIds(i);
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
    Int32ConstArrayView ghost_ids = m_sync_info->receiveInfo().localIds(i);
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
  MultiDataSynchronizeBuffer buffer(tm, m_sync_info.get(), m_buffer_copier);

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
        ARCANE_FATAL("Variable '{0}' can not be synchronized because it is not a numeric data", var->name());
      MutableMemoryView mem_view = numapi->memoryView();
      all_datatype_size += mem_view.datatypeSize();
      buffer.setDataView(index, mem_view);
      ++index;
    }
  }

  _setCurrentDevice();

  // TODO: à passer en paramètre.
  bool is_compare_sync = false;
  buffer.prepareSynchronize(all_datatype_size, is_compare_sync);

  m_implementation_instance->setDataSynchronizeInfo(m_sync_info.get());
  m_implementation_instance->compute();
  m_implementation_instance->beginSynchronize(&buffer);
  m_implementation_instance->endSynchronize(&buffer);
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
  ARCANE_CHECK_POINTER(vs_buf);
  IParallelMng* pm = m_parallel_mng;

  const bool use_blocking_send = false;
  Int32 nb_message = vs_buf->nbRank();

  /*pm->traceMng()->info() << " ** ** COMMON BEGIN SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/

  // Envoie les messages de réception non bloquant
  for (Integer i = 0; i < nb_message; ++i) {
    Int32 target_rank = vs_buf->targetRank(i);
    auto buf = _toLegacySmallView(vs_buf->receiveBuffer(i));
    if (!buf.empty()) {
      Parallel::Request rval = pm->recv(buf, target_rank, false);
      m_all_requests.add(rval);
    }
  }

  vs_buf->copyAllSend();

  // Envoie les messages d'envoi en mode non bloquant.
  for (Integer i = 0; i < nb_message; ++i) {
    Int32 target_rank = vs_buf->targetRank(i);
    auto buf = _toLegacySmallView(vs_buf->sendBuffer(i));

    //ConstArrayView<SimpleType> const_share = share_local_buffer;
    if (!buf.empty()) {
      //for( Integer i=0, is=share_local_buffer.size(); i<is; ++i )
      //trace->info() << "TO rank=" << vsi.m_target_rank << " I=" << i << " V=" << share_local_buffer[i]
      //                << " lid=" << share_grp[i] << " v2=" << var_values[share_grp[i]];
      Parallel::Request rval = pm->send(buf, target_rank, use_blocking_send);
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
