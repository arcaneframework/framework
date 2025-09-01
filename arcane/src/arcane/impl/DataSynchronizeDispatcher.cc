// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeDispatcher.cc                                (C) 2000-2025 */
/*                                                                           */
/* Gestion de la synchronisation d'une instance de 'IData'.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/IDataSynchronizeDispatcher.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/internal/MemoryBuffer.h"

#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IData.h"
#include "arcane/core/internal/IDataInternal.h"

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

class DataSynchronizeDispatcherBase
{
 public:

  explicit DataSynchronizeDispatcherBase(const DataSynchronizeDispatcherBuildInfo& bi);
  ~DataSynchronizeDispatcherBase();

 protected:

  IParallelMng* m_parallel_mng = nullptr;
  Runner* m_runner = nullptr;
  Ref<DataSynchronizeInfo> m_sync_info;
  Ref<IDataSynchronizeImplementation> m_synchronize_implementation;

 protected:

  void _compute();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeDispatcherBase::
DataSynchronizeDispatcherBase(const DataSynchronizeDispatcherBuildInfo& bi)
: m_parallel_mng(bi.parallelMng())
, m_sync_info(bi.synchronizeInfo())
, m_synchronize_implementation(bi.synchronizeImplementation())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeDispatcherBase::
~DataSynchronizeDispatcherBase()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notifie l'implémentation que les informations de synchronisation
 * ont changé.
 */
void DataSynchronizeDispatcherBase::
_compute()
{
  m_synchronize_implementation->compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de la synchronisation pour une donnée.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeDispatcher
: private ReferenceCounterImpl
, public DataSynchronizeDispatcherBase
, public IDataSynchronizeDispatcher
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  explicit DataSynchronizeDispatcher(const DataSynchronizeDispatcherBuildInfo& bi)
  : DataSynchronizeDispatcherBase(bi)
  , m_sync_buffer(bi.parallelMng()->traceMng(), m_sync_info.get(), bi.bufferCopier())
  {
  }

 public:

  void compute() override { _compute(); }
  void setSynchronizeBuffer(Ref<MemoryBuffer> buffer) override { m_sync_buffer.setSynchronizeBuffer(buffer); }
  void beginSynchronize(INumericDataInternal* data, bool is_compare_sync) override;
  DataSynchronizeResult endSynchronize() override;

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

void DataSynchronizeDispatcher::
beginSynchronize(INumericDataInternal* data, bool is_compare_sync)
{
  ARCANE_CHECK_POINTER(data);

  MutableMemoryView mem_view = data->memoryView();

  if (m_is_in_sync)
    ARCANE_FATAL("_beginSynchronize() has already been called");
  m_is_in_sync = true;

  m_is_empty_sync = (mem_view.bytes().size() == 0);
  if (m_is_empty_sync)
    return;
  m_sync_buffer.setDataView(mem_view);
  m_sync_buffer.prepareSynchronize(is_compare_sync);
  m_synchronize_implementation->beginSynchronize(&m_sync_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeResult DataSynchronizeDispatcher::
endSynchronize()
{
  if (!m_is_in_sync)
    ARCANE_FATAL("No pending synchronize(). You need to call beginSynchronize() before");
  DataSynchronizeResult result;
  if (!m_is_empty_sync) {
    m_synchronize_implementation->endSynchronize(&m_sync_buffer);
    result = m_sync_buffer.finalizeSynchronize();
  }
  m_is_in_sync = false;
  return result;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataSynchronizeDispatcher> IDataSynchronizeDispatcher::
create(const DataSynchronizeDispatcherBuildInfo& build_info)
{
  return makeRef<IDataSynchronizeDispatcher>(new DataSynchronizeDispatcher(build_info));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Synchronisation d'une liste de variables.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeMultiDispatcher
: public IDataSynchronizeMultiDispatcher
{
 public:

  explicit DataSynchronizeMultiDispatcher(const DataSynchronizeDispatcherBuildInfo& bi)
  : m_parallel_mng(bi.parallelMng())
  , m_sync_info(bi.synchronizeInfo())
  {
  }

  void compute() override {}
  void setSynchronizeBuffer(Ref<MemoryBuffer>) override {}
  void synchronize(ConstArrayView<IVariable*> vars) override;

 private:

  IParallelMng* m_parallel_mng = nullptr;
  Ref<DataSynchronizeInfo> m_sync_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataSynchronizeMultiDispatcher::
synchronize(ConstArrayView<IVariable*> vars)
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
    for (IVariable* var : vars) {
      var->serialize(sbuf, share_ids, nullptr);
    }
    sbuf->allocateBuffer();
    sbuf->setMode(ISerializer::ModePut);
    for (IVariable* var : vars) {
      var->serialize(sbuf, share_ids, nullptr);
    }
  }
  exchanger->processExchange();
  for (Integer i = 0; i < nb_rank; ++i) {
    ISerializeMessage* msg = exchanger->messageToReceive(i);
    ISerializer* sbuf = msg->serializer();
    Int32ConstArrayView ghost_ids = m_sync_info->receiveInfo().localIds(i);
    sbuf->setMode(ISerializer::ModeGet);
    for (IVariable* var : vars) {
      var->serialize(sbuf, ghost_ids, nullptr);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Synchronisation d'une liste de variables.
 *
 * \brief Version 2 qui utilise directement des buffers au lieu
 * d'un ISerializer.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeMultiDispatcherV2
: public DataSynchronizeDispatcherBase
, public IDataSynchronizeMultiDispatcher
{
 public:

  explicit DataSynchronizeMultiDispatcherV2(const DataSynchronizeDispatcherBuildInfo& bi)
  : DataSynchronizeDispatcherBase(bi)
  , m_sync_buffer(bi.parallelMng()->traceMng(), m_sync_info.get(), bi.bufferCopier())
  {
  }

  void compute() override { _compute(); }
  void setSynchronizeBuffer(Ref<MemoryBuffer> buffer) override { m_sync_buffer.setSynchronizeBuffer(buffer); }
  void synchronize(ConstArrayView<IVariable*> vars) override;

 private:

  MultiDataSynchronizeBuffer m_sync_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataSynchronizeMultiDispatcherV2::
synchronize(ConstArrayView<IVariable*> vars)
{
  const Int32 nb_var = vars.size();
  m_sync_buffer.setNbData(nb_var);

  // Récupère les emplacements mémoire des données des variables et leur taille
  {
    Int32 index = 0;
    for (IVariable* var : vars) {
      INumericDataInternal* numapi = var->data()->_commonInternal()->numericData();
      if (!numapi)
        ARCANE_FATAL("Variable '{0}' can not be synchronized because it is not a numeric data", var->name());
      MutableMemoryView mem_view = numapi->memoryView();
      m_sync_buffer.setDataView(index, mem_view);
      ++index;
    }
  }

  // TODO: à passer en paramètre de la fonction
  bool is_compare_sync = false;
  m_sync_buffer.prepareSynchronize(is_compare_sync);

  m_synchronize_implementation->beginSynchronize(&m_sync_buffer);
  m_synchronize_implementation->endSynchronize(&m_sync_buffer);
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
class SimpleDataSynchronizeImplementation
: public AbstractDataSynchronizeImplementation
{
 public:

  class Factory;
  explicit SimpleDataSynchronizeImplementation(Factory* f);

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

class SimpleDataSynchronizeImplementation::Factory
: public IDataSynchronizeImplementationFactory
{
 public:

  explicit Factory(IParallelMng* pm)
  : m_parallel_mng(pm)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new SimpleDataSynchronizeImplementation(this);
    return makeRef<IDataSynchronizeImplementation>(x);
  }

 public:

  IParallelMng* m_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleDataSynchronizeImplementation::
SimpleDataSynchronizeImplementation(Factory* f)
: m_parallel_mng(f->m_parallel_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateSimpleVariableSynchronizerFactory(IParallelMng* pm)
{
  auto* x = new SimpleDataSynchronizeImplementation::Factory(pm);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleDataSynchronizeImplementation::
beginSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  ARCANE_CHECK_POINTER(vs_buf);
  IParallelMng* pm = m_parallel_mng;

  const bool use_blocking_send = false;
  Int32 nb_message = vs_buf->nbRank();

  /*pm->traceMng()->info() << " ** ** COMMON BEGIN SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_info=" << &this->m_sync_info;*/

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

void SimpleDataSynchronizeImplementation::
endSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  IParallelMng* pm = m_parallel_mng;

  /*pm->traceMng()->info() << " ** ** COMMON END SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_info=" << &this->m_sync_info;*/

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

IDataSynchronizeMultiDispatcher* IDataSynchronizeMultiDispatcher::
create(const DataSynchronizeDispatcherBuildInfo& bi)
{
  // TODO: Une fois qu'on aura supprimer l'ancien mécanisme, il faudra
  // modifier l'API ne pas utiliser 'VariableCollection' mais une liste
  // de \a INumericDataInternal
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_LEGACY_MULTISYNCHRONIZE", true))
    if (v.value() >= 1)
      return new DataSynchronizeMultiDispatcher(bi);
  return new DataSynchronizeMultiDispatcherV2(bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
