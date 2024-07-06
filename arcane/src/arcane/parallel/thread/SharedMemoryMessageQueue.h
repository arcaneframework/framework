// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMessageQueue.h                                  (C) 2000-2024 */
/*                                                                           */
/* Implémentation d'une file de messages en mémoire partagée.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_SHAREDMEMORYMESSAGEQUEUE_H
#define ARCANE_PARALLEL_THREAD_SHAREDMEMORYMESSAGEQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/parallel/thread/ISharedMemoryMessageQueue.h"

#include "arcane/ISerializer.h"
#include "arcane/Parallel.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryParallelMng;
class SharedMemoryMessageRequest;
class SharedMemoryMessageQueue;
using MessageRank = MP::MessageRank;
using MessageTag = MP::MessageTag;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief File de messages entre les rangs partagés par un SharedMemoryParallelMng.
 */
class ARCANE_THREAD_EXPORT SharedMemoryMessageQueue
: public ISharedMemoryMessageQueue
{
 public:

  class SubQueue;

 public:

  SharedMemoryMessageQueue() : m_nb_thread(0), m_atomic_request_id(0){}
  ~SharedMemoryMessageQueue() override;

 public:

  void init(Int32 nb_thread) override;
  void waitAll(ArrayView<Request> requests) override;
  void waitSome(Int32 rank,ArrayView<Request> requests,ArrayView<bool> requests_done,
                bool is_non_blockign) override;
  void setTraceMng(Int32 rank,ITraceMng* tm) override;

 public:

  Request addReceive(const PointToPointMessageInfo& message,ReceiveBufferInfo buf) override;
  Request addSend(const PointToPointMessageInfo& message,SendBufferInfo buf) override;

 public:

  MessageId probe(const PointToPointMessageInfo& message) override;
  MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) override;

 private:

  Int32 m_nb_thread = 0;
  UniqueArray<SubQueue*> m_sub_queues;
  std::atomic<Int64> m_atomic_request_id;

 private:

  SubQueue* _getSubQueue(MessageRank rank)
  {
    return m_sub_queues[rank.value()];
  }
  Int64 _getNextRequestId()
  {
    return m_atomic_request_id.fetch_add(1);
  }
  Request _request(SharedMemoryMessageRequest* tmr);
  SubQueue* _getSourceSubQueue(const MP::PointToPointMessageInfo& message);
  SubQueue* _getDestinationSubQueue(const MP::PointToPointMessageInfo& message);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Message entre SharedMemoryMessageQueue.
 *
 * Cette classe gère à la fois les messages d'envoi et de réception.
 */
class ARCANE_THREAD_EXPORT SharedMemoryMessageRequest
{
 public:
  class SortFunctor
  {
   public:
    explicit SortFunctor(Int32 nb_thread) : m_nb_thread(nb_thread){}
    bool operator()(SharedMemoryMessageRequest* r1,SharedMemoryMessageRequest* r2) const
    {
      if (!r1){
        if (!r2)
          return true;
        else
          return false;
      }
      if (!r2)
        return true;
      if (r1->isRecv() && !r2->isRecv())
        return true;
      if (!r1->isRecv() && r2->isRecv())
        return false;
      Int32 i1 = _getQueueIndex(r1->orig(),r1->dest());
      Int32 i2 = _getQueueIndex(r2->orig(),r2->dest());
      if (i1==i2)
        return r1->id() < r2->id();
      return i1 < i2;
    }
    Int32 _getQueueIndex(MessageRank thread1,MessageRank thread2) const
    {
      if (thread1.isNull())
        ARCANE_FATAL("Null rank for thread1");
      if (thread2.isNull())
        ARCANE_FATAL("Null rank for thread2");
      // TODO: gérer dest()==A_NULL_RANK.
      return thread1.value() + (thread2.value() * m_nb_thread);
    }
    Int32 m_nb_thread;
  };
 public:
  using SubQueue = SharedMemoryMessageQueue::SubQueue;
 public:
  //! Créé une requête d'envoie
  SharedMemoryMessageRequest(SubQueue* queue,Int64 request_id,MessageRank orig,
                             MessageRank dest,MessageTag tag,ReceiveBufferInfo buf)
  : m_queue(queue), m_request_id(request_id), m_is_recv(true)
  , m_orig(orig), m_dest(dest), m_tag(tag), m_receive_buffer_info(buf)
  {
  }
  //! Créé une requête de réception
  SharedMemoryMessageRequest(SubQueue* queue,Int64 request_id,MessageRank orig,
                             MessageRank dest,MessageTag tag,SendBufferInfo buf)
  : m_queue(queue), m_request_id(request_id), m_is_recv(false)
  , m_orig(orig), m_dest(dest), m_tag(tag), m_send_buffer_info(buf)
  {
  }
 public:
  MessageRank orig() { return m_orig; }
  MessageRank dest() { return m_dest; }
  MessageTag tag() { return m_tag; }
  bool isRecv() { return m_is_recv; }
  bool isDone() { return m_is_done; }
  void setDone(bool v) { m_is_done = v; }
  SendBufferInfo sendBufferInfo() { return m_send_buffer_info; }
  ReceiveBufferInfo receiveBufferInfo() { return m_receive_buffer_info; }
  SharedMemoryMessageQueue::SubQueue* queue() { return m_queue; }
  void copyFromSender(SharedMemoryMessageRequest* sender);
  Int64 id() const { return m_request_id; }
  void destroy();
  ISerializer* recvSerializer() { return m_receive_buffer_info.serializer(); }
  const ISerializer* sendSerializer() { return m_send_buffer_info.serializer(); }
  // Dans le cas ou dest()==A_NULL_RANK, positionne une fois le message recu le rang d'origine.
  void setSource(MessageRank s)
  {
    if (isRecv())
      m_dest = s;
  }
  //! Requête associée dans le cas où c'est un `receive` issu d'un `probe`
  SharedMemoryMessageRequest* matchingSendRequest() { return m_matching_send_request; }
  void setMatchingSendRequest(SharedMemoryMessageRequest* r) { m_matching_send_request = r; }

 private:
  SubQueue* m_queue;
  Int64 m_request_id;
  bool m_is_recv;
  MessageRank m_orig;
  MessageRank m_dest;
  MessageTag m_tag;
  bool m_is_done = false;
  SharedMemoryMessageRequest* m_matching_send_request = nullptr;
  bool m_is_destroyed = false;
  SendBufferInfo m_send_buffer_info;
  ReceiveBufferInfo m_receive_buffer_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

