// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngDispatcher.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire du parallélisme sur un domaine.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLELMNGDISPATCHER_H
#define ARCANE_PARALLELMNGDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IParallelMng.h"
#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/message_passing/MessagePassingMng.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class IParallelDispatchT;
class ITimeStats;
namespace MP = ::Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT ParallelMngDispatcherBuildInfo
{
 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use overload with Ref<MP::MessagePassingMng> and Ref<MP::Dispatchers> instead")
  ParallelMngDispatcherBuildInfo(MP::Dispatchers* dispatchers,
                                 MP::MessagePassingMng* mpm);
  ParallelMngDispatcherBuildInfo(Ref<MP::Dispatchers> dispatchers,
                                 Ref<MP::MessagePassingMng> mpm);
  ParallelMngDispatcherBuildInfo(Int32 comm_rank, Int32 comm_size);

 public:

  Int32 commRank() const { return m_comm_rank; }
  Int32 commSize() const { return m_comm_size; }
  Ref<MP::Dispatchers> dispatchersRef() const { return m_dispatchers_ref; }
  Ref<MP::MessagePassingMng> messagePassingMngRef() const { return m_message_passing_mng_ref; }

  ARCANE_DEPRECATED_REASON("Y2022: Use messagePassingMngRef() instead")
  MP::MessagePassingMng* messagePassingMng() const { return m_message_passing_mng; }
  ARCANE_DEPRECATED_REASON("Y2022: Use dispatchersRef() instead")
  MP::Dispatchers* dispatchers() const { return m_dispatchers; }

 private:

  Int32 m_comm_rank;
  Int32 m_comm_size;
  MP::Dispatchers* m_dispatchers;
  Ref<MP::Dispatchers> m_dispatchers_ref;
  MP::MessagePassingMng* m_message_passing_mng;
  Ref<MP::MessagePassingMng> m_message_passing_mng_ref;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Redirige la gestion des messages des sous-domaines
 * suivant le type de l'argument.
 */
class ARCANE_CORE_EXPORT ParallelMngDispatcher
: public ReferenceCounterImpl
, public IParallelMng
{
 public:

  //! Implémentation de Arccore::MessagePassing::IControlDispatcher.
  class ARCANE_CORE_EXPORT DefaultControlDispatcher
  : public MP::IControlDispatcher
  {
   public:

    explicit DefaultControlDispatcher(IParallelMng* pm);

  public:

    void waitAllRequests(ArrayView<Request> requests) override;
    void waitSomeRequests(ArrayView<Request> requests,
                          ArrayView<bool> indexes, bool is_non_blocking) override;
    IMessagePassingMng* commSplit(bool keep) override;
    void barrier() override;
    Request nonBlockingBarrier() override;
    MessageId probe(const PointToPointMessageInfo& message) override;
    MP::MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) override;
    Ref<Parallel::IRequestList> createRequestListRef() override;
    MP::IProfiler* profiler() const override { return nullptr; }
    void setProfiler(MP::IProfiler* p) override;

  private:

    IParallelMng* m_parallel_mng;
  };

  //! Implémentation de Arccore::MessagePassing::ISerializeDispatcher.
  class SerializeDispatcher;

  friend class ParallelMngInternal;

 public:

  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  explicit ParallelMngDispatcher(const ParallelMngDispatcherBuildInfo& bi);
  ~ParallelMngDispatcher() override;

 public:

  ParallelMngDispatcher(const ParallelMngDispatcher&) = delete;
  ParallelMngDispatcher(ParallelMngDispatcher&&) = delete;
  ParallelMngDispatcher& operator=(ParallelMngDispatcher&&) = delete;
  ParallelMngDispatcher& operator=(const ParallelMngDispatcher&) = delete;

 private:

  void _setArccoreDispatchers();

 public:

  IMessagePassingMng* messagePassingMng() const override;
  void broadcastString(String& str, Int32 rank) override;
  void broadcastMemoryBuffer(ByteArray& bytes, Int32 rank) override;

  //! Redéfinit ici allGather pour éviter de cacher le symbole dans les classes dérivées.
  void allGather(ISerializer* send_serializer, ISerializer* recv_serializer) override;

 public:

#define ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(field, type) \
 public:\
  void allGather(ConstArrayView<type> send_buf, ArrayView<type> recv_buf) override; \
  void gather(ConstArrayView<type> send_buf, ArrayView<type> recv_buf, Integer rank) override; \
  void allGatherVariable(ConstArrayView<type> send_buf, Array<type>& recv_buf) override; \
  void gatherVariable(ConstArrayView<type> send_buf, Array<type>& recv_buf, Integer rank) override; \
  void scatterVariable(ConstArrayView<type> send_buf, ArrayView<type> recv_buf, Integer root) override; \
  type reduce(eReduceType rt, type v) override; \
  void reduce(eReduceType rt, ArrayView<type> v) override;\
  void broadcast(ArrayView<type> send_buf, Integer id) override;\
  void send(ConstArrayView<type> values, Integer id) override;\
  void recv(ArrayView<type> values, Integer id) override;\
  Request send(ConstArrayView<type> values, Int32 rank, bool is_blocked) override; \
  Request send(Span<const type> values, const PointToPointMessageInfo& message) override; \
  Request recv(ArrayView<type> values, Int32 rank, bool is_blocked) override;\
  Request receive(Span<type> values, const PointToPointMessageInfo& message) override; \
  void sendRecv(ConstArrayView<type> send_buf, ArrayView<type> recv_buf, Integer id) override;\
  void allToAll(ConstArrayView<type> send_buf, ArrayView<type> recv_buf, Integer count) override;\
  void allToAllVariable(ConstArrayView<type> send_buf, Int32ConstArrayView send_count,\
                        Int32ConstArrayView send_index, ArrayView<type> recv_buf,\
                        Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) override;\
  type scan(eReduceType rt, type v);                                     \
  void computeMinMaxSum(type val, type& min_val, type& max_val, type& sum_val, Int32& min_proc, Int32& max_proc) override;\
  void computeMinMaxSum(ConstArrayView<type> values,\
                        ArrayView<type> min_values,\
                        ArrayView<type> max_values,\
                        ArrayView<type> sum_values,\
                        ArrayView<Int32> min_ranks,\
                        ArrayView<Int32> max_ranks) override;\
  void scan(eReduceType rt, ArrayView<type> v) override;\
 protected:\
  IParallelDispatchT<type>* field;

  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_char, char)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_unsigned_char, unsigned char)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_signed_char, signed char)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_short, short)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_unsigned_short, unsigned short)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_int, int)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_unsigned_int, unsigned int)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_long, long)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_unsigned_long, unsigned long)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_long_long, long long)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_unsigned_long_long, unsigned long long)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_float, float)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_double, double)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_long_double, long double)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_apreal, APReal)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_real2, Real2)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_real3, Real3)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_real2x2, Real2x2)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_real3x3, Real3x3)
  ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE(m_hpreal, HPReal)

#undef ARCANE_PARALLEL_MANAGER_DISPATCH_PROTOTYPE

 public:

  virtual IParallelDispatchT<char>* dispatcher(char*);
  virtual IParallelDispatchT<signed char>* dispatcher(signed char*);
  virtual IParallelDispatchT<unsigned char>* dispatcher(unsigned char*);
  virtual IParallelDispatchT<short>* dispatcher(short*);
  virtual IParallelDispatchT<unsigned short>* dispatcher(unsigned short*);
  virtual IParallelDispatchT<int>* dispatcher(int*);
  virtual IParallelDispatchT<unsigned int>* dispatcher(unsigned int*);
  virtual IParallelDispatchT<long>* dispatcher(long*);
  virtual IParallelDispatchT<unsigned long>* dispatcher(unsigned long*);
  virtual IParallelDispatchT<long long>* dispatcher(long long*);
  virtual IParallelDispatchT<unsigned long long>* dispatcher(unsigned long long*);
  virtual IParallelDispatchT<APReal>* dispatcher(APReal*);
  virtual IParallelDispatchT<float>* dispatcher(float*);
  virtual IParallelDispatchT<double>* dispatcher(double*);
  virtual IParallelDispatchT<long double>* dispatcher(long double*);
  virtual IParallelDispatchT<Real2>* dispatcher(Real2*);
  virtual IParallelDispatchT<Real3>* dispatcher(Real3*);
  virtual IParallelDispatchT<Real2x2>* dispatcher(Real2x2*);
  virtual IParallelDispatchT<Real3x3>* dispatcher(Real3x3*);
  virtual IParallelDispatchT<HPReal>* dispatcher(HPReal*);

 public:

  template <class CreatorType>  void
  createDispatchers(CreatorType& ct)
  {
    m_char = ct.template create<char>();
    m_signed_char = ct.template create<signed char>();
    m_unsigned_char = ct.template create<unsigned char>();
    m_short = ct.template create<short>();
    m_unsigned_short = ct.template create<unsigned short>();
    m_int = ct.template create<int>();
    m_unsigned_int = ct.template create<unsigned int>();
    m_long = ct.template create<long>();
    m_unsigned_long = ct.template create<unsigned long>();
    m_long_long = ct.template create<long long>();
    m_unsigned_long_long = ct.template create<unsigned long long>();

    m_float = ct.template create<float>();
    m_double = ct.template create<double>();
    m_long_double = ct.template create<long double>();

    m_apreal = ct.template create<APReal>();
    m_real2 = ct.template create<Real2>();
    m_real3 = ct.template create<Real3>();
    m_real2x2 = ct.template create<Real2x2>();
    m_real3x3 = ct.template create<Real3x3>();
    m_hpreal = ct.template create<HPReal>();

    _setArccoreDispatchers();
  }

 public:

  ITimeStats* timeStats() const override { return m_time_stats; }
  void setTimeStats(ITimeStats* ts) override;
  ITimeMetricCollector* timeMetricCollector() const override;

  UniqueArray<Integer> waitSomeRequests(ArrayView<Request> requests) override;
  UniqueArray<Integer> testSomeRequests(ArrayView<Request> requests) override;
  void processMessages(ConstArrayView<ISerializeMessage*> messages) override;
  void processMessages(ConstArrayView<Ref<ISerializeMessage>> messages) override;
  ISerializeMessageList* createSerializeMessageList() final;
  Ref<ISerializeMessageList> createSerializeMessageListRef() final;
  IParallelMng* createSubParallelMng(Int32ConstArrayView kept_ranks) final;
  Ref<IParallelMng> createSubParallelMngRef(Int32ConstArrayView kept_ranks) override;

 public:

  IParallelMngInternal* _internalApi() override { return m_parallel_mng_internal; }

 protected:

  MP::MessagePassingMng* _messagePassingMng() const { return m_message_passing_mng_ref.get(); }
  UniqueArray<Integer> _doWaitRequests(ArrayView<Request> requests,Parallel::eWaitType wait_type);
  virtual ISerializeMessageList* _createSerializeMessageList() =0;
  virtual IParallelMng* _createSubParallelMng(Int32ConstArrayView kept_ranks) =0;
  virtual bool _isAcceleratorAware() const { return false; }
  virtual Ref<IParallelMng> _createSubParallelMngRef(Int32 color, Int32 key);

 protected:

  TimeMetricAction _communicationTimeMetricAction() const;
  void _setControlDispatcher(MP::IControlDispatcher* d);
  void _setSerializeDispatcher(MP::ISerializeDispatcher* d);

 private:
  
  ITimeStats* m_time_stats = nullptr;
  Ref<MP::Dispatchers> m_mp_dispatchers_ref;
  Ref<MP::MessagePassingMng> m_message_passing_mng_ref;
  MP::IControlDispatcher* m_control_dispatcher = nullptr;
  MP::ISerializeDispatcher* m_serialize_dispatcher = nullptr;
  IParallelMngInternal* m_parallel_mng_internal = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelMngContainerBase
: public ReferenceCounterImpl
, public IParallelMngContainer
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
