// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SequentialParallelMng.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Gestion du parallélisme dans le cas séquentiel.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include "arccore/message_passing/PointToPointMessageInfo.h"

#include "arcane/utils/Collection.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/NullThreadMng.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/IIOMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IParallelDispatch.h"
#include "arcane/core/ParallelMngDispatcher.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/Timer.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/internal/SerializeMessage.h"
#include "arcane/core/internal/ParallelMngInternal.h"

#include "arcane/parallel/IStat.h"

#include "arcane/impl/TimerMng.h"
#include "arcane/impl/GetVariablesValuesParallelOperation.h"
#include "arcane/impl/ParallelExchanger.h"
#include "arcane/impl/ParallelTopology.h"
#include "arcane/impl/ParallelReplication.h"
#include "arcane/impl/SequentialParallelSuperMng.h"
#include "arcane/impl/SequentialParallelMng.h"
#include "arcane/impl/internal/ParallelMngUtilsFactoryBase.h"
#include "arcane/impl/internal/VariableSynchronizer.h"

#include "arccore/message_passing/RequestListBase.h"
#include "arccore/message_passing/SerializeMessageList.h"
#include "arccore/message_passing/internal/IMachineMemoryWindowBaseInternal.h"
#include "arccore/message_passing/internal/IDynamicMachineMemoryWindowBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using RequestListBase = Arcane::MessagePassing::internal::RequestListBase;
using IRequestList = Parallel::IRequestList;
using namespace Arcane::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IVariableSynchronizer*
createNullVariableSynchronizer(IParallelMng* pm,const ItemGroup& group);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SequentialRequestList
: public RequestListBase
{
 public:
  void _wait(Parallel::eWaitType wait_mode)
  {
    ARCANE_UNUSED(wait_mode);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface des messages pour le type \a Type
 */
template<class Type>
class SequentialParallelDispatchT
: public TraceAccessor
, public IParallelDispatchT<Type>
, public ITypeDispatcher<Type>
{
 public:
  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;
 public:
  SequentialParallelDispatchT(ITraceMng* tm)
  : TraceAccessor(tm) {}
  void finalize() override {}
 public:
  void broadcast(ArrayView<Type> send_buf,Int32 rank) override
  {
    ARCANE_UNUSED(send_buf);
    ARCANE_UNUSED(rank);
  }
  void broadcast(Span<Type> send_buf,Int32 rank) override
  {
    ARCANE_UNUSED(send_buf);
    ARCANE_UNUSED(rank);
  }
  void allGather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf) override
  {
    recv_buf.copy(send_buf);
  }
  void allGather(Span<const Type> send_buf,Span<Type> recv_buf) override
  {
    recv_buf.copy(send_buf);
  }
  void gather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    recv_buf.copy(send_buf);
  }
  void gather(Span<const Type> send_buf,Span<Type> recv_buf,Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    recv_buf.copy(send_buf);
  }
  void scatterVariable(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Int32 root) override
  {
    ARCANE_UNUSED(root);
    recv_buf.copy(send_buf);
  }
  void scatterVariable(Span<const Type> send_buf,Span<Type> recv_buf,Int32 root) override
  {
    ARCANE_UNUSED(root);
    recv_buf.copy(send_buf);
  }
  void allGatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf) override
  {
    gatherVariable(send_buf,recv_buf,0);
  }
  void allGatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf) override
  {
    gatherVariable(send_buf,recv_buf,0);
  }
  void gatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf,Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    recv_buf.resize(send_buf.size());
    ArrayView<Type> av(recv_buf);
    av.copy(send_buf);
  }
  void gatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf,Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    recv_buf.resize(send_buf.size());
    Span<Type> av(recv_buf.span());
    av.copy(send_buf);
  }
  void allToAll(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer count) override
  {
    ARCANE_UNUSED(count);
    recv_buf.copy(send_buf);
  }
  void allToAll(Span<const Type> send_buf,Span<Type> recv_buf,Int32 count) override
  {
    ARCANE_UNUSED(count);
    recv_buf.copy(send_buf);
  }
  void allToAllVariable(ConstArrayView<Type> send_buf,
                        Int32ConstArrayView send_count,
                        Int32ConstArrayView send_index,
                        ArrayView<Type> recv_buf,
                        Int32ConstArrayView recv_count,
                        Int32ConstArrayView recv_index) override
  {
    ARCANE_UNUSED(send_count);
    ARCANE_UNUSED(recv_count);
    ARCANE_UNUSED(send_index);
    ARCANE_UNUSED(recv_index);
    recv_buf.copy(send_buf);
  }
  void allToAllVariable(Span<const Type> send_buf,
                        ConstArrayView<Int32> send_count,
                        ConstArrayView<Int32> send_index,
                        Span<Type> recv_buf,
                        ConstArrayView<Int32> recv_count,
                        ConstArrayView<Int32> recv_index) override
  {
    ARCANE_UNUSED(send_count);
    ARCANE_UNUSED(recv_count);
    ARCANE_UNUSED(send_index);
    ARCANE_UNUSED(recv_index);
    recv_buf.copy(send_buf);
  }
  Request send(ConstArrayView<Type> send_buffer,Int32 rank,bool is_blocked) override
  {
    return send(Span<const Type>(send_buffer),rank,is_blocked);
  }
  Request send(Span<const Type> send_buffer,Int32 rank,bool is_blocked) override
  {
    ARCANE_UNUSED(send_buffer);
    ARCANE_UNUSED(rank);
    if (is_blocked)
      throw NotSupportedException(A_FUNCINFO,"blocking send is not allowed in sequential");
    return Request();
  }
  Request send(Span<const Type> send_buffer,const PointToPointMessageInfo& message) override
  {
    ARCANE_UNUSED(send_buffer);
    if (message.isBlocking())
      throw NotSupportedException(A_FUNCINFO,"blocking send is not allowed in sequential");
    return Request();
  }
  Request recv(ArrayView<Type> recv_buffer,Int32 rank,bool is_blocked) override
  {
    return receive(Span<Type>(recv_buffer),rank,is_blocked);
  }
  void send(ConstArrayView<Type> send_buffer,Int32 rank) override
  {
    ARCANE_UNUSED(send_buffer);
    ARCANE_UNUSED(rank);
    throw NotSupportedException(A_FUNCINFO,"send is not allowed in sequential");
  }
  void recv(ArrayView<Type> recv_buffer,Int32 rank) override
  {
    ARCANE_UNUSED(recv_buffer);
    ARCANE_UNUSED(rank);
    throw NotSupportedException(A_FUNCINFO,"recv is not allowed in sequential");
  }
  Request receive(Span<Type> recv_buffer,Int32 rank,bool is_blocked) override
  {
    ARCANE_UNUSED(recv_buffer);
    ARCANE_UNUSED(rank);
    if (is_blocked)
      throw NotSupportedException(A_FUNCINFO,"blocking receive is not allowed in sequential");
    return Request();
  }
  Request receive(Span<Type> recv_buffer,const PointToPointMessageInfo& message) override
  {
    ARCANE_UNUSED(recv_buffer);
    if (message.isBlocking())
      throw NotSupportedException(A_FUNCINFO,"blocking receive is not allowed in sequential");
    return Request();
  }
  void sendRecv(ConstArrayView<Type> send_buffer,ArrayView<Type> recv_buffer,Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    recv_buffer.copy(send_buffer);
  }
  Type allReduce(eReduceType op,Type v) override
  {
    ARCANE_UNUSED(op);
    return v;
  }
  void allReduce(eReduceType op,ArrayView<Type> send_buf) override
  {
    ARCANE_UNUSED(op);
    ARCANE_UNUSED(send_buf);
  }
  void allReduce(eReduceType op,Span<Type> send_buf) override
  {
    ARCANE_UNUSED(op);
    ARCANE_UNUSED(send_buf);
  }
  Request nonBlockingAllReduce(eReduceType op,Span<const Type> send_buf,Span<Type> recv_buf) override
  {
    ARCANE_UNUSED(op);
    ARCANE_UNUSED(send_buf);
    ARCANE_UNUSED(recv_buf);
    return Request();
  }
  Request nonBlockingAllGather(Span<const Type> send_buf, Span<Type> recv_buf) override
  {
    recv_buf.copy(send_buf);
    return Request();
  }
  Request nonBlockingBroadcast(Span<Type> send_buf, Int32 rank) override
  {
    ARCANE_UNUSED(send_buf);
    ARCANE_UNUSED(rank);
    return Request();
  }
  Request nonBlockingGather(Span<const Type> send_buf, Span<Type> recv_buf, Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    recv_buf.copy(send_buf);
    return Request();
  }
  Request nonBlockingAllToAll(Span<const Type> send_buf, Span<Type> recv_buf, Int32 count) override
  {
    ARCANE_UNUSED(count);
    recv_buf.copy(send_buf);
    return Request();
  }
  Request nonBlockingAllToAllVariable(Span<const Type> send_buf, ConstArrayView<Int32> send_count,
                                      ConstArrayView<Int32> send_index, Span<Type> recv_buf,
                                      ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index) override
  {
    ARCANE_UNUSED(send_count);
    ARCANE_UNUSED(recv_count);
    ARCANE_UNUSED(send_index);
    ARCANE_UNUSED(recv_index);
    recv_buf.copy(send_buf);
    return Request();
  }
  Type scan(eReduceType op,Type v) override
  {
    ARCANE_UNUSED(op);
    return v;
  }
  void scan(eReduceType op,ArrayView<Type> send_buf) override
  {
    ARCANE_UNUSED(op);
    ARCANE_UNUSED(send_buf);
  }
  void computeMinMaxSum(Type val,Type& min_val,Type& max_val,Type& sum_val,
                        Int32& min_rank,
                        Int32& max_rank) override
  {
    min_val = max_val = sum_val = val;
    min_rank = max_rank = 0;
  }
  void computeMinMaxSum(ConstArrayView<Type> values,
                        ArrayView<Type> min_values,
                        ArrayView<Type> max_values,
                        ArrayView<Type> sum_values,
                        ArrayView<Int32> min_ranks,
                        ArrayView<Int32> max_ranks) override
  {
    const Integer n = values.size();
    for(Integer i=0;i<n;++i) {
      min_values[i] = max_values[i] = sum_values[i] = values[i];
      min_ranks[i] = max_ranks[i] = 0;
    }
  }
  ITypeDispatcher<Type>* toArccoreDispatcher() override { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SequentialMachineMemoryWindowBaseInternal
: public IMachineMemoryWindowBaseInternal
{
 public:

  SequentialMachineMemoryWindowBaseInternal(Int64 sizeof_segment, Int32 sizeof_type)
  : m_sizeof_segment(sizeof_segment)
  , m_max_sizeof_segment(sizeof_segment)
  , m_sizeof_type(sizeof_type)
  , m_segment(sizeof_segment)
  {}

  ~SequentialMachineMemoryWindowBaseInternal() override = default;

 public:

  Int32 sizeofOneElem() const override
  {
    return m_sizeof_type;
  }

  Span<std::byte> segmentView() override
  {
    return m_segment.span().subSpan(0, m_sizeof_segment);
  }
  Span<std::byte> segmentView(const Int32 rank) override
  {
    if (rank != 0) {
      ARCANE_FATAL("Rank {0} is unavailable (Sequential)", rank);
    }
    return m_segment.span().subSpan(0, m_sizeof_segment);
  }
  Span<std::byte> windowView() override
  {
    return m_segment.span().subSpan(0, m_sizeof_segment);
  }

  Span<const std::byte> segmentConstView() const override
  {
    return m_segment.constSpan().subSpan(0, m_sizeof_segment);
  }
  Span<const std::byte> segmentConstView(const Int32 rank) const override
  {
    if (rank != 0) {
      ARCANE_FATAL("Rank {0} is unavailable (Sequential)", rank);
    }
    return m_segment.constSpan().subSpan(0, m_sizeof_segment);
  }
  Span<const std::byte> windowConstView() const override
  {
    return m_segment.constSpan().subSpan(0, m_sizeof_segment);
  }

  void resizeSegment(const Int64 new_sizeof_segment) override
  {
    if (new_sizeof_segment > m_max_sizeof_segment) {
      ARCANE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_sizeof_segment = new_sizeof_segment;
  }

  ConstArrayView<Int32> machineRanks() const override
  {
    return ConstArrayView<Int32>{ 1, &m_my_rank };
  }

  void barrier() const override {}

 private:

  Int64 m_sizeof_segment = 0;
  Int64 m_max_sizeof_segment = 0;

  Int32 m_sizeof_type = 0;
  UniqueArray<std::byte> m_segment;
  Int32 m_my_rank = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SequentialDynamicMachineMemoryWindowBaseInternal
: public IDynamicMachineMemoryWindowBaseInternal
{
 public:

  SequentialDynamicMachineMemoryWindowBaseInternal(Int64 sizeof_segment, Int32 sizeof_type)
  : m_sizeof_type(sizeof_type)
  , m_segment(sizeof_segment)
  {}
  ~SequentialDynamicMachineMemoryWindowBaseInternal() override = default;

 public:

  Int32 sizeofOneElem() const override
  {
    return m_sizeof_type;
  }
  ConstArrayView<Int32> machineRanks() const override
  {
    return ConstArrayView<Int32>{ 1, &m_my_rank };
  }
  void barrier() const override {}

  Span<std::byte> segmentView() override
  {
    return m_segment;
  }
  Span<std::byte> segmentView(Int32 rank) override
  {
    if (rank != 0) {
      ARCANE_FATAL("Rank {0} is unavailable (Sequential)", rank);
    }
    return m_segment;
  }
  Span<const std::byte> segmentConstView() const override
  {
    return m_segment;
  }
  Span<const std::byte> segmentConstView(Int32 rank) const override
  {
    if (rank != 0) {
      ARCANE_FATAL("Rank {0} is unavailable (Sequential)", rank);
    }
    return m_segment;
  }
  void add(Span<const std::byte> elem) override
  {
    if (elem.size() % m_sizeof_type != 0) {
      ARCCORE_FATAL("Sizeof elem not valid");
    }
    m_segment.addRange(elem);
  }
  void add() override {}
  void addToAnotherSegment(Int32 rank, Span<const std::byte> elem) override
  {
    if (rank != 0) {
      ARCANE_FATAL("Rank {0} is unavailable (Sequential)", rank);
    }
    if (elem.size() % m_sizeof_type != 0) {
      ARCCORE_FATAL("Sizeof elem not valid");
    }
    m_segment.addRange(elem);
  }
  void addToAnotherSegment() override {}

  void reserve(Int64 new_capacity) override
  {
    m_segment.reserve(new_capacity);
  }
  void reserve() override {}
  void resize(Int64 new_size) override
  {
    m_segment.resize(new_size);
  }
  void resize() override {}
  void shrink() override
  {
    m_segment.shrink();
  }

 private:

  Int32 m_sizeof_type;
  UniqueArray<std::byte> m_segment;
  Int32 m_my_rank = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SequentialParallelMngUtilsFactory
: public ParallelMngUtilsFactoryBase
{
 public:
  Ref<ITransferValuesParallelOperation> createTransferValuesOperation(IParallelMng*) override
  {
    throw NotImplementedException(A_FUNCINFO);
  }
  Ref<IVariableSynchronizer> createSynchronizer(IParallelMng* pm,IItemFamily* family) override
  {
    return makeRef(createNullVariableSynchronizer(pm,family->allItems()));
  }
  Ref<IVariableSynchronizer> createSynchronizer(IParallelMng* pm,const ItemGroup& group) override
  {
    return makeRef(createNullVariableSynchronizer(pm,group));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire du parallélisme en mode séquentiel.
 *
 * En mode séquentiel, le parallélisme n'existe pas. Ce gestionnaire ne
 * fait donc rien.
*/
class SequentialParallelMng
: public ParallelMngDispatcher
{
 public:

  class Impl;

 private:
  // Construit un gestionnaire séquentiel.
  SequentialParallelMng(const SequentialParallelMngBuildInfo& bi);
 public:
  ~SequentialParallelMng();

  bool isParallel() const override { return false; }
  Int32 commRank() const override { return 0; }
  Int32 commSize() const override { return 1; }
  void* getMPICommunicator() override { return m_communicator.communicatorAddress(); }
  Parallel::Communicator communicator() const override { return m_communicator; }
  Parallel::Communicator machineCommunicator() const override { return m_communicator; }
  bool isThreadImplementation() const override { return false; }
  bool isHybridImplementation() const override { return false; }
  void setBaseObject(IBase* m);
  ITraceMng* traceMng() const override { return m_trace.get(); }
  IThreadMng* threadMng() const override { return m_thread_mng; }
  ITimerMng* timerMng() const override { return m_timer_mng; }
  IParallelMng* worldParallelMng() const override { return m_world_parallel_mng; }
  IIOMng* ioMng() const override { return m_io_mng; }

  void initialize() override ;
  bool isMasterIO() const override { return true; }
  Int32 masterIORank() const override { return 0; }

 public:

  void allGather(ISerializer* send_serializer, ISerializer* recv_serializer) override
  {
    recv_serializer->copy(send_serializer);
  }
  void sendSerializer(ISerializer* values,Int32 rank) override
  {
    ARCANE_UNUSED(values);
    ARCANE_UNUSED(rank);
  }
  Request sendSerializer(ISerializer* values,Int32 rank,ByteArray& bytes) override
  {
    ARCANE_UNUSED(values);
    ARCANE_UNUSED(rank);
    ARCANE_UNUSED(bytes);
    return Parallel::Request();
  }
  ISerializeMessage* createSendSerializer(Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    return new SerializeMessage(0,0,ISerializeMessage::MT_Send);
  }
  void recvSerializer(ISerializer* values,Int32 rank) override
  {
    ARCANE_UNUSED(values);
    ARCANE_UNUSED(rank);
  }
  ISerializeMessage* createReceiveSerializer(Int32 rank) override
  {
    ARCANE_UNUSED(rank);
    return new SerializeMessage(0,0,ISerializeMessage::MT_Recv);
  }

  void broadcastString(String& str,Int32 rank) override
  {
    ARCANE_UNUSED(str);
    ARCANE_UNUSED(rank);
  }
  void broadcastMemoryBuffer(ByteArray& bytes,Int32 rank) override
  {
    ARCANE_UNUSED(bytes);
    ARCANE_UNUSED(rank);
  }
  void broadcastSerializer(ISerializer* values,Int32 rank) override
  {
    ARCANE_UNUSED(values);
    ARCANE_UNUSED(rank);
  }
  MessageId probe(const PointToPointMessageInfo& message) override
  {
    ARCANE_UNUSED(message);
    return MessageId();
  }
  MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) override
  {
    ARCANE_UNUSED(message);
    return MessageSourceInfo();
  }
  Request sendSerializer(const ISerializer* values,const PointToPointMessageInfo& message) override
  {
    ARCANE_UNUSED(values);
    ARCANE_UNUSED(message);
    return Parallel::Request();
  }
  Request receiveSerializer(ISerializer* values,const PointToPointMessageInfo& message) override
  {
    ARCANE_UNUSED(values);
    ARCANE_UNUSED(message);
    return Parallel::Request();
  }

  void processMessages(ConstArrayView<ISerializeMessage*> messages) override
  {
    ARCANE_UNUSED(messages);
  }
  void freeRequests(ArrayView<Parallel::Request> requests) override
  {
    ARCANE_UNUSED(requests);
  }

  void printStats() override {}
  void barrier() override {}

  IParallelMng* sequentialParallelMng() override { return this; }
  Ref<IParallelMng> sequentialParallelMngRef() override { return makeRef<IParallelMng>(this); }

  void waitAllRequests(ArrayView<Request> requests) override
  {
    ARCANE_UNUSED(requests);
  }

  // Les requetes sont forcement deja satisfaites.
  UniqueArray<Integer> waitSomeRequests(ArrayView<Request> requests) override
  {
    ARCANE_UNUSED(requests);
	  return UniqueArray<Integer>();
  }
  UniqueArray<Integer> testSomeRequests(ArrayView<Request> requests) override
  {
	  return waitSomeRequests(requests);
  }

  ISerializeMessageList* _createSerializeMessageList() override
  {
    return new Arccore::MessagePassing::internal::SerializeMessageList(messagePassingMng());
  }
  Real reduceRank(eReduceType rt,Real v,Int32* rank)
  {
    Real rv = reduce(rt,v);
    if (rank)
      *rank = 0;
    return rv;
  }
  IGetVariablesValuesParallelOperation* createGetVariablesValuesOperation() override
  {
    return new GetVariablesValuesParallelOperation(this);
  }
  ITransferValuesParallelOperation* createTransferValuesOperation() override
  {
    throw NotImplementedException(A_FUNCINFO);
  }
  IParallelExchanger* createExchanger() override
  {
    return new ParallelExchanger(this);
  }
  IVariableSynchronizer* createSynchronizer(IItemFamily* family) override
  {
    return createNullVariableSynchronizer(this,family->allItems());
  }
  IVariableSynchronizer* createSynchronizer(const ItemGroup& group) override
  {
    return createNullVariableSynchronizer(this,group);
  }
  IParallelTopology* createTopology() override
  {
    ParallelTopology* t = new ParallelTopology(this);
    t->initialize();
    return t;
  }

  IParallelReplication* replication() const override
  {
    return m_replication;
  }

  void setReplication(IParallelReplication* v) override
  {
    delete m_replication;
    m_replication = v;
  }

  Ref<IRequestList> createRequestListRef() override
  {
    IRequestList* r = new SequentialRequestList();
    return makeRef(r);
  }

  Ref<IParallelMngUtilsFactory> _internalUtilsFactory() const override
  {
    return m_utils_factory;
  }

  Parallel::IStat* stat() override
  {
    return m_stat;
  }

  void build() override;

  IParallelNonBlockingCollective* nonBlockingCollective() const override { return 0; }

  IParallelMngInternal* _internalApi() override { return m_parallel_mng_internal; }

 public:
  
  static IParallelMng* create(const SequentialParallelMngBuildInfo& bi)
  {
    if (!bi.traceMng())
      ARCANE_THROW(ArgumentException,"null traceMng()");
    if (!bi.threadMng())
      ARCANE_THROW(ArgumentException,"null threadMng()");
    auto x = new SequentialParallelMng(bi);
    x->build();
    return x;
  }
  static Ref<IParallelMng> createRef(const SequentialParallelMngBuildInfo& bi)
  {
    return makeRef(create(bi));
  }

 protected:

  IParallelMng* _createSubParallelMng(Int32ConstArrayView kept_ranks) override
  {
    ARCANE_UNUSED(kept_ranks);
    SequentialParallelMngBuildInfo bi(m_timer_mng,m_world_parallel_mng);
    bi.setThreadMng(m_thread_mng);
    bi.setTraceMng(m_trace.get());
    bi.setCommunicator(m_communicator);
    return create(bi);
  }

 private:

  ReferenceCounter<ITraceMng> m_trace;
  IThreadMng* m_thread_mng = nullptr;
  ITimerMng* m_timer_mng = nullptr;
  IParallelMng* m_world_parallel_mng = nullptr;
  IIOMng* m_io_mng;
  Parallel::IStat* m_stat;
  IParallelReplication* m_replication;
  MP::Communicator m_communicator;
  Ref<IParallelMngUtilsFactory> m_utils_factory;
  IParallelMngInternal* m_parallel_mng_internal = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IIOMng*
arcaneCreateIOMng(IParallelMng* psm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IParallelMng*
arcaneCreateSequentialParallelMng(const SequentialParallelMngBuildInfo& bi)
{
  return SequentialParallelMng::create(bi);
}
extern "C++" Ref<IParallelMng>
arcaneCreateSequentialParallelMngRef(const SequentialParallelMngBuildInfo& bi)
{
  return SequentialParallelMng::createRef(bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SequentialParallelMng::Impl
: public ParallelMngInternal
{
 public:

  explicit Impl(SequentialParallelMng* pm)
  : ParallelMngInternal(pm)
  {}

  ~Impl() override = default;

 public:

  Ref<IMachineMemoryWindowBaseInternal> createMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(new SequentialMachineMemoryWindowBaseInternal(sizeof_segment, sizeof_type));
  }

  Ref<IDynamicMachineMemoryWindowBaseInternal> createDynamicMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(new SequentialDynamicMachineMemoryWindowBaseInternal(sizeof_segment, sizeof_type));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SequentialParallelMng::
SequentialParallelMng(const SequentialParallelMngBuildInfo& bi)
: ParallelMngDispatcher(ParallelMngDispatcherBuildInfo(0,1))
, m_trace(bi.traceMng())
, m_thread_mng(bi.threadMng())
, m_timer_mng(bi.m_timer_mng)
, m_world_parallel_mng(bi.m_world_parallel_mng)
, m_io_mng(nullptr)
, m_stat(nullptr)
, m_replication(new ParallelReplication())
, m_communicator(bi.communicator())
, m_utils_factory(makeRef<IParallelMngUtilsFactory>(new SequentialParallelMngUtilsFactory()))
, m_parallel_mng_internal(new Impl(this))
{
  ARCANE_CHECK_PTR(m_trace);
  ARCANE_CHECK_PTR(m_thread_mng);
  if (!m_world_parallel_mng)
    m_world_parallel_mng = this;
  m_stat = Parallel::createDefaultStat();
  _messagePassingMng()->setCommunicator(m_communicator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SequentialParallelMng::
~SequentialParallelMng()
{
  delete m_parallel_mng_internal;
  delete m_stat;
  delete m_replication;
  delete m_io_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialParallelMng::
setBaseObject(IBase* sd)
{
  ARCANE_UNUSED(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
// Classe pour créer les différents dispatchers
class DispatchCreator
{
 public:
  DispatchCreator(ITraceMng* tm) : m_tm(tm){}
 public:
  template<typename DataType> SequentialParallelDispatchT<DataType>*
  create()
  {
    return new SequentialParallelDispatchT<DataType>(m_tm);
  }
  ITraceMng* m_tm;
};
}

void SequentialParallelMng::
build()
{
  m_io_mng = arcaneCreateIOMng(this);
  DispatchCreator creator(m_trace.get());
  this->createDispatchers(creator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialParallelMng::
initialize()
{
  traceMng()->info() << "** ** MPI Communicator = " << getMPICommunicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Construit un superviseur séquentiel lié au superviseur \a sm
SequentialParallelSuperMng::
SequentialParallelSuperMng(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_application(sbi.application())
, m_thread_mng(new NullThreadMng())
, m_timer_mng(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Construit un superviseur séquentiel lié au superviseur \a sm
SequentialParallelSuperMng::
SequentialParallelSuperMng(const ServiceBuildInfo& sbi,Parallel::Communicator comm)
: AbstractService(sbi)
, m_application(sbi.application())
, m_thread_mng(new NullThreadMng())
, m_timer_mng(nullptr)
, m_communicator(comm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Construit un superviseur séquentiel lié au superviseur \a sm
SequentialParallelSuperMng::
~SequentialParallelSuperMng()
{
  delete m_thread_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialParallelSuperMng::
initialize()
{
  IApplication* app = m_application;
  ITraceMng* tm = app->traceMng();
  SequentialParallelDispatchT<Byte>* c = new SequentialParallelDispatchT<Byte>(tm);
  SequentialParallelDispatchT<Int32>* i32 = new SequentialParallelDispatchT<Int32>(tm);
  SequentialParallelDispatchT<Int64>* i64 = new SequentialParallelDispatchT<Int64>(tm);
  SequentialParallelDispatchT<Real>* r = new SequentialParallelDispatchT<Real>(tm);
  _setDispatchers(c,i32,i64,r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialParallelSuperMng::
build()
{
  if (!m_timer_mng){
    m_owned_timer_mng = new TimerMng(traceMng());
    m_timer_mng = m_owned_timer_mng.get();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> SequentialParallelSuperMng::
internalCreateWorldParallelMng(Int32 local_rank)
{
  if (local_rank!=0)
    ARCANE_THROW(ArgumentException,"Bad local_rank '{0}' (should be 0)",local_rank);

  SequentialParallelMngBuildInfo bi(m_timer_mng,nullptr);
  bi.setThreadMng(threadMng());
  bi.setTraceMng(traceMng());
  bi.setCommunicator(communicator());
  return SequentialParallelMng::createRef(bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialParallelSuperMng::
tryAbort()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(SequentialParallelSuperMng,IParallelSuperMng,
                                    SequentialParallelSuperMng);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SequentialParallelMngBuilder
: public ParallelMngContainerBase
{
 public:
  SequentialParallelMngBuilder(IApplication* app,Parallel::Communicator comm)
  : m_application(app), m_thread_mng(new NullThreadMng()),
    m_timer_mng(new TimerMng(app->traceMng())), m_communicator(comm){}
  ~SequentialParallelMngBuilder() override
  {
    delete m_timer_mng;
    delete m_thread_mng;
  }

 public:

  void build() {}
  Ref<IParallelMng> _createParallelMng(Int32 local_rank,ITraceMng* tm) override;

 public:
  IApplication* m_application;
  IThreadMng* m_thread_mng;
  ITimerMng* m_timer_mng;
  Parallel::Communicator m_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> SequentialParallelMngBuilder::
_createParallelMng(Int32 local_rank,ITraceMng* tm)
{
  if (local_rank!=0)
    ARCANE_THROW(ArgumentException,"Bad local_rank '{0}' (should be 0)",local_rank);

  SequentialParallelMngBuildInfo bi(m_timer_mng,nullptr);
  bi.setTraceMng(tm);
  bi.setThreadMng(m_thread_mng);
  bi.setCommunicator(m_communicator);
  return arcaneCreateSequentialParallelMngRef(bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SequentialParallelMngContainerFactory
: public AbstractService
, public IParallelMngContainerFactory
{
 public:
  SequentialParallelMngContainerFactory(const ServiceBuildInfo& sbi)
  : AbstractService(sbi), m_application(sbi.application()){}
 public:
  Ref<IParallelMngContainer>
  _createParallelMngBuilder(Int32 nb_rank, Parallel::Communicator comm, Parallel::Communicator machine_comm) override
  {
    ARCANE_UNUSED(nb_rank);
    ARCANE_UNUSED(machine_comm);
    auto x = new SequentialParallelMngBuilder(m_application,comm);
    x->build();
    return makeRef<IParallelMngContainer>(x);
  }
 private:
  IApplication* m_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SequentialParallelMngContainerFactory,
                        ServiceProperty("SequentialParallelMngContainerFactory",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IParallelMngContainerFactory));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

