// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMessageQueue.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Implementation of a shared memory message queue.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"

#include "arccore/common/internal/MemoryResourceMng.h"

#include "arcane/parallel/thread/SharedMemoryMessageQueue.h"
#include "arcane/parallel/thread/IAsyncQueue.h"

#include "arcane/core/ISerializeMessage.h"

// Macro to display messages for debugging
#define TRACE_DEBUG(format_str, ...) \
  if (m_is_debug && m_trace_mng) { \
    m_trace_mng->info() << String::format(format_str, __VA_ARGS__); \
    m_trace_mng->flush(); \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Copies the sender message information into the receive message.
 *
 * If an 'ISerializer' is available, it uses it. Otherwise, it is a
 * memory area and the values are copied directly.
 *
 * \note It would be possible, if we knew the origin of the memory area,
 * to avoid copying by just passing the pointer.
 */
void SharedMemoryMessageRequest::
copyFromSender(SharedMemoryMessageRequest* sender)
{
  SendBufferInfo send_info = sender->sendBufferInfo();
  ReceiveBufferInfo receive_info = this->receiveBufferInfo();

  const ISerializer* send_serializer = send_info.serializer();
  ISerializer* receive_serializer = receive_info.serializer();
  if (receive_serializer) {
    if (!send_serializer)
      ARCANE_FATAL("No send serializer for receive serializer");
    receive_serializer->copy(send_serializer);
    return;
  }

  ByteConstSpan send_span = send_info.memoryBuffer();
  ByteSpan receive_span = receive_info.memoryBuffer();
  Int64 send_size = send_span.size();
  Int64 receive_size = receive_span.size();
  if (send_size > receive_size)
    ARCANE_FATAL("Not enough memory for receiving message receive={0} send={1}",
                 receive_size, send_size);

  MemoryResourceMng::genericCopy(ConstMemoryView(send_span), MutableMemoryView(receive_span));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageRequest::
destroy()
{
  if (m_is_destroyed)
    ARCANE_FATAL("Request already destroyed");
  m_is_destroyed = true;
  // Comment out for debug.
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \a Asynchronous queue.
 */
class RequestAsyncQueue
{
 public:

  RequestAsyncQueue()
  : m_async_queue(IAsyncQueue::createQueue())
  {}
  ~RequestAsyncQueue()
  {
    delete m_async_queue;
  }

 public:

  void push(SharedMemoryMessageRequest* v)
  {
    m_async_queue->push(v);
  }
  SharedMemoryMessageRequest* pop()
  {
    return reinterpret_cast<SharedMemoryMessageRequest*>(m_async_queue->pop());
  }
  SharedMemoryMessageRequest* tryPop()
  {
    return reinterpret_cast<SharedMemoryMessageRequest*>(m_async_queue->tryPop());
  }

 private:

  IAsyncQueue* m_async_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief File for messages from a rank in shared memory.
 *
 * Uses an asynchronous queue to store messages.
 *
 * \note The methods of this class are not thread-safe, which means that two
 * different threads cannot post messages from the same rank without prior
 * synchronization. Making this class thread-safe would allow for behavior
 * similar to MPI in MPI_THREAD_MULTIPLE mode.
 */
class SharedMemoryMessageQueue::SubQueue
{
  static MessageTag SERIALIZER_TAG() { return MessageTag(125); }

 public:

  SubQueue(SharedMemoryMessageQueue* master_queue, MessageRank rank);

 public:

  MessageRank rank() const { return m_rank; }
  void setTraceMng(ITraceMng* tm) { m_trace_mng = tm; }
  void wait(SharedMemoryMessageRequest* tmr);
  void testRequest(SharedMemoryMessageRequest* tmr);
  void waitRequestAvailable();
  void checkRequestAvailable();
  void waitSome(ArrayView<Request> requests, ArrayView<bool> requests_done, bool is_non_blocking);
  MessageId probe(const PointToPointMessageInfo& message);
  MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message);

 public:

  SharedMemoryMessageRequest*
  addReceive(Int64 request_id, const PointToPointMessageInfo& message, ReceiveBufferInfo buf);
  SharedMemoryMessageRequest*
  addSend(Int64 request_id, const PointToPointMessageInfo& message, SendBufferInfo buf);

 private:

  SharedMemoryMessageQueue* m_master_queue = nullptr;
  MessageRank m_rank;
  UniqueArray<SharedMemoryMessageRequest*> m_send_requests;
  UniqueArray<SharedMemoryMessageRequest*> m_recv_requests;
  UniqueArray<SharedMemoryMessageRequest*> m_done_requests;
  RequestAsyncQueue m_async_message_queue;
  ITraceMng* m_trace_mng = nullptr;
  bool m_is_debug = false;
  bool m_is_allow_null_rank_for_any_source = true;

 private:

  void _removeRequest(SharedMemoryMessageRequest* tmr, Array<SharedMemoryMessageRequest*>& requests);
  bool _checkSendDone(SharedMemoryMessageRequest* tmr_send);
  bool _checkRecvDone(SharedMemoryMessageRequest* tmr_recv);
  void _checkRequestDone(SharedMemoryMessageRequest* tmr);
  void _cleanupRequestIfDone(SharedMemoryMessageRequest* tmr);
  SharedMemoryMessageRequest*
  _getMatchingSendRequest(MessageRank recv_dest, MessageRank recv_orig, MessageTag tag);
  void _testOrWaitRequestAvailable(bool is_blocking);
  SharedMemoryMessageRequest*
  _createReceiveRequest(Int64 request_id, MessageRank dest, MessageTag tag,
                        ReceiveBufferInfo receive_buffer);
  SharedMemoryMessageRequest*
  _createSendRequest(Int64 request_id, MessageRank orig, MessageTag tag,
                     SendBufferInfo send_buffer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageQueue::SubQueue::
SubQueue(SharedMemoryMessageQueue* master_queue, MessageRank rank)
: m_master_queue(master_queue)
, m_rank(rank)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCCORE_ALLOW_NULL_RANK_FOR_MPI_ANY_SOURCE", true))
    m_is_allow_null_rank_for_any_source = v.value() != 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageRequest* SharedMemoryMessageQueue::SubQueue::
_createReceiveRequest(Int64 request_id, MessageRank dest,
                      MessageTag tag, ReceiveBufferInfo receive_buffer)
{
  auto* tmr = new SharedMemoryMessageRequest(this, request_id, m_rank, dest, tag, receive_buffer);
  return tmr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageRequest* SharedMemoryMessageQueue::SubQueue::
_createSendRequest(Int64 request_id, MessageRank orig,
                   MessageTag tag, SendBufferInfo send_buffer)
{
  SubQueue* queue = m_master_queue->_getSubQueue(orig);
  auto* tmr = new SharedMemoryMessageRequest(queue, request_id, orig, m_rank, tag, send_buffer);
  return tmr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageRequest* SharedMemoryMessageQueue::SubQueue::
addReceive(Int64 request_id, const PointToPointMessageInfo& message, ReceiveBufferInfo buf)
{
  SharedMemoryMessageRequest* tmr = nullptr;
  if (message.isRankTag()) {
    MessageTag tag = message.tag();
    MessageRank dest = message.destinationRank();
    tmr = _createReceiveRequest(request_id, dest, tag, buf);
    TRACE_DEBUG("** ADD RECV queue={0} id={1} ORIG={2} DEST={3} tag={4} tmr={5} size={6} serializer={7}",
                this, request_id, m_rank, dest, tag, tmr, buf.memoryBuffer().size(), buf.serializer());
  }
  else if (message.isMessageId()) {
    MessageId message_id = message.messageId();
    MessageId::SourceInfo si = message_id.sourceInfo();
    MessageRank dest = si.rank();
    MessageTag tag = si.tag();

    // We know the 'send' request that matches this one. We must therefore
    // position it in \a tmr now to ensure that the correct one is used.
    // To do this, search for the send corresponding to the ID of our request
    Int64 req_id = (size_t)message_id;
    SharedMemoryMessageRequest* send_request = nullptr;
    for (Integer i = 0, n = m_send_requests.size(); i < n; ++i)
      if (m_send_requests[i]->id() == req_id) {
        send_request = m_send_requests[i];
        break;
      }
    if (!send_request)
      ARCANE_FATAL("Can not find matching send request from MessageId");

    tmr = _createReceiveRequest(request_id, dest, tag, buf);
    TRACE_DEBUG("** ADD RECV FromMessageId queue={0} id={1} ORIG={2} DEST={3} tag={4} tmr={5} size={6} serializer={7}",
                this, request_id, m_rank, dest, tag, tmr, buf.memoryBuffer().size(), buf.serializer());
    tmr->setMatchingSendRequest(send_request);
  }
  else
    ARCANE_THROW(NotSupportedException, "Invalid 'MessageInfo'");

  m_recv_requests.add(tmr);
  return tmr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageRequest* SharedMemoryMessageQueue::SubQueue::
addSend(Int64 request_id, const PointToPointMessageInfo& message, SendBufferInfo buf)
{
  MessageTag tag = message.tag();
  if (tag.isNull())
    ARCANE_THROW(ArgumentException, "null tag");
  MessageRank orig = message.emiterRank();
  auto* tmr = _createSendRequest(request_id, orig, tag, buf);
  m_async_message_queue.push(tmr);
  TRACE_DEBUG("** ADD SEND queue={0} ORIG={1} DEST={2} tag={3} size={4} tmr={5} serializer={6}",
              this, orig, m_rank, tag, buf.memoryBuffer().size(), tmr, buf.serializer());
  return tmr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
_checkRequestDone(SharedMemoryMessageRequest* tmr)
{
  if (tmr->isDone())
    ARCANE_FATAL("Can not check already done request");

  if (tmr->isRecv())
    _checkRecvDone(tmr);
  else
    _checkSendDone(tmr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Cleans up the request \a tmr if it is finished.
 *
 * If the request is completed, it is removed from the list of requests.
 */
void SharedMemoryMessageQueue::SubQueue::
_cleanupRequestIfDone(SharedMemoryMessageRequest* tmr)
{
  if (tmr->isDone()) {
    if (tmr->isRecv())
      _removeRequest(tmr, m_recv_requests);
    else
      _removeRequest(tmr, m_done_requests);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
_testOrWaitRequestAvailable(bool is_blocking)
{
  SharedMemoryMessageRequest* sq = nullptr;
  if (is_blocking)
    sq = m_async_message_queue.pop();
  else
    sq = m_async_message_queue.tryPop();
  if (sq) {
    if (sq->orig() == m_rank)
      m_done_requests.add(sq);
    else
      m_send_requests.add(sq);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
waitRequestAvailable()
{
  // Blocks until a message is received.
  _testOrWaitRequestAvailable(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
checkRequestAvailable()
{
  _testOrWaitRequestAvailable(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
wait(SharedMemoryMessageRequest* tmr)
{
  if (tmr->queue() != this)
    ARCANE_FATAL("Bad queue");
  TRACE_DEBUG("**** WAIT MESSAGE tmr={0} rank={1} recv?={2} dest={3}"
              " nb_send={4} nb_done={5}",
              tmr, m_rank, tmr->isRecv(),
              tmr->dest(), m_send_requests.size(), m_done_requests.size());
  while (!tmr->isDone()) {
    _checkRequestDone(tmr);

    if (!tmr->isDone()) {
      // Blocks until a message is received.
      waitRequestAvailable();
    }
  }

  _cleanupRequestIfDone(tmr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
testRequest(SharedMemoryMessageRequest* tmr)
{
  if (tmr->queue() != this)
    ARCANE_FATAL("Bad queue");

  _checkRequestDone(tmr);
  _cleanupRequestIfDone(tmr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
_removeRequest(SharedMemoryMessageRequest* tmr, Array<SharedMemoryMessageRequest*>& requests)
{
  for (Integer i = 0, n = requests.size(); i < n; ++i) {
    SharedMemoryMessageRequest* tmr2 = requests[i];
    if (tmr == tmr2) {
      requests.remove(i);
      return;
    }
  }
  ARCANE_FATAL("Can not remove request");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SharedMemoryMessageQueue::SubQueue::
_checkSendDone(SharedMemoryMessageRequest* tmr_send)
{
  for (SharedMemoryMessageRequest* tmr : m_done_requests) {
    if (tmr == tmr_send) {
      tmr_send->setDone(true);
      return true;
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Checks if a send message matches the request \a tmr_recv.
 *
 * Checks if a request in the list of sent messages matches
 * \a tmr_recv. This is the case if the source/destination pair is suitable
 * or if the destination is A_NULL_RANK (which corresponds to MPI_ANY_SOURCE
 * in the MPI case).
 * It is important to process requests in order of arrival to comply with the MPI standard.
 *
 * \retval true if a matching request was found.
 * \retval false otherwise.
 */
SharedMemoryMessageRequest* SharedMemoryMessageQueue::SubQueue::
_getMatchingSendRequest(MessageRank recv_dest, MessageRank recv_orig, MessageTag tag)
{
  bool is_any_tag = tag.isNull();
  bool is_any_dest = recv_dest.isNull() || recv_dest.isAnySource();
  if (recv_dest.isNull() && !m_is_allow_null_rank_for_any_source)
    ARCANE_FATAL("Can not use probe() with null rank. Use MessageRank::anySourceRank() instead");
  for (Integer j = 0, n = m_send_requests.size(); j < n; ++j) {
    SharedMemoryMessageRequest* tmr_send = m_send_requests[j];
    TRACE_DEBUG("CHECK RECV DONE id={7} tmr_send={0} recv_dest={1}"
                " recv_orig={2} send_dest={3} send_orig={4} request={5}/{6}\n",
                tmr_send, recv_dest, recv_orig,
                tmr_send->dest(), tmr_send->orig(), j, n, m_rank);
    if (recv_orig == tmr_send->dest()) {
      bool is_rank_ok = (recv_dest == tmr_send->orig()) || (is_any_dest && m_rank == recv_orig);
      bool is_tag_ok = (is_any_tag || tmr_send->tag() == tag);
      if (is_rank_ok && is_tag_ok) {
        return tmr_send;
      }
    }
  }
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SharedMemoryMessageQueue::SubQueue::
_checkRecvDone(SharedMemoryMessageRequest* tmr_recv)
{
  // Checks if the send is already associated with our request. This is the case
  // if we used a probe() call.
  auto* tmr_send = tmr_recv->matchingSendRequest();
  if (!tmr_send)
    tmr_send = _getMatchingSendRequest(tmr_recv->dest(), tmr_recv->orig(), tmr_recv->tag());
  if (tmr_send) {
    tmr_recv->setSource(tmr_send->orig());
    tmr_recv->copyFromSender(tmr_send);
    tmr_recv->setDone(true);
    tmr_send->queue()->m_async_message_queue.push(tmr_send);
    _removeRequest(tmr_send, m_send_requests);
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::SubQueue::
waitSome(ArrayView<Request> requests, ArrayView<bool> requests_done,
         bool is_non_blocking)
{
  Integer nb_request = requests.size();
  requests_done.fill(false);
  bool one_request_done = false;
  TRACE_DEBUG("** WAIT SOME REQUEST rank={0} nb_request={1}\n", m_rank, nb_request);
  while (!one_request_done) {
    if (is_non_blocking)
      checkRequestAvailable();
    bool has_valid_request = false;
    for (Integer i = 0; i < nb_request; ++i) {
      Request request = requests[i];
      if (!request.isValid())
        continue;
      SharedMemoryMessageRequest* tmr = requests[i];
      if (!tmr)
        continue;
      this->testRequest(tmr);
      if (tmr->isDone()) {
        one_request_done = true;
        requests_done[i] = true;
        tmr->destroy();
        if (requests[i].hasSubRequest())
          ARCANE_THROW(NotImplementedException, "handling of sub requests");
        requests[i].reset();
      }
      has_valid_request = true;
    }
    // If at least one request is completed, exit the loop.
    if (one_request_done)
      break;
    // If not, no request succeeded. We block until there is
    // a message in the queue unless there are no valid requests
    // (This is possible if all requests in the list are null requests).
    if (!has_valid_request)
      break;
    // In non-blocking mode, we already tested at the beginning of the loop if
    // a request is available. If we are here, we can exit directly
    // otherwise we will block.
    if (is_non_blocking)
      break;
    TRACE_DEBUG("** WAIT REQUEST AVAILABLE rank={0}", m_rank);
    waitRequestAvailable();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MP::MessageId SharedMemoryMessageQueue::SubQueue::
probe(const MP::PointToPointMessageInfo& message)
{
  TRACE_DEBUG("Probe rank={0} nb_send={1} nb_receive={2} queue={3} is_valid={4}",
              m_rank, m_send_requests.size(), m_recv_requests.size(), this, message.isValid());
  if (!message.isValid())
    return MessageId();

  // The message must be initialized with a (rank/tag) pair.
  if (!message.isRankTag())
    ARCCORE_FATAL("Invalid message_info: message.isRankTag() is false");

  MessageRank rank = message.destinationRank();
  MessageTag tag = message.tag();
  bool is_blocking = message.isBlocking();
  if (is_blocking)
    ARCANE_THROW(NotImplementedException, "blocking probe");

  // TODO: look into adding anti-loop safety
  // TODO: we should check that if this
  // method is called twice with the same information, the same message is not retrieved.
  // When that is the case, legacyProbe() will need to be modified accordingly.
  for (;;) {
    _testOrWaitRequestAvailable(is_blocking);
    auto* req = _getMatchingSendRequest(rank, m_rank, tag);
    if (req) {
      // TODO: Verify that the request has a buffer and not an
      // 'ISerializer'.
      Int64 send_size = req->sendBufferInfo().memoryBuffer().size();
      MessageId::SourceInfo si(req->orig(), req->tag(), send_size);
      return MessageId(si, (size_t)req->id());
    }
    if (!is_blocking)
      // In non-blocking mode, exit the loop even if no request is found.
      break;
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MP::MessageSourceInfo SharedMemoryMessageQueue::SubQueue::
legacyProbe(const MP::PointToPointMessageInfo& message)
{
  // Performs a normal probe but does not retain message information.
  // NOTE: this works because probe() can return the same
  // message multiple times. When that is no longer the case, this will need to be modified.
  MP::MessageId message_id = probe(message);
  if (message_id.isValid())
    return message_id.sourceInfo();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageQueue::
~SharedMemoryMessageQueue()
{
  for (SubQueue* sq : m_sub_queues)
    delete sq;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::
init(Integer nb_thread)
{
  m_atomic_request_id = 1;
  m_nb_thread = nb_thread;
  Integer nb_queue = nb_thread;
  m_sub_queues.resize(nb_queue);
  for (Integer i = 0; i < nb_queue; ++i) {
    m_sub_queues[i] = new SubQueue(this, MessageRank(i));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageQueue::SubQueue* SharedMemoryMessageQueue::
_getSourceSubQueue(const MP::PointToPointMessageInfo& message)
{
  MessageRank orig = message.emiterRank();
  if (orig.isNull())
    ARCANE_THROW(ArgumentException, "null message.sourceRank()");
  return _getSubQueue(orig);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMessageQueue::SubQueue* SharedMemoryMessageQueue::
_getDestinationSubQueue(const MP::PointToPointMessageInfo& message)
{
  MessageRank dest = message.destinationRank();
  if (dest.isNull())
    ARCANE_THROW(ArgumentException, "null message.destinationRank()");
  return _getSubQueue(dest);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::
waitAll(ArrayView<Request> requests)
{
  Integer nb_request = requests.size();
  // To prevent blocking, the requests must be processed
  // in ascending order of queues, and 'receive' FIRST!!
  UniqueArray<SharedMemoryMessageRequest*> sorted_requests(nb_request);
  for (Integer i = 0; i < nb_request; ++i) {
    sorted_requests[i] = (SharedMemoryMessageRequest*)requests[i].requestAsVoidPtr();
  }
  std::sort(std::begin(sorted_requests), std::end(sorted_requests),
            SharedMemoryMessageRequest::SortFunctor(m_nb_thread));

#if 0
  for( Integer i=0; i<nb_request; ++i ){
    SharedMemoryMessageRequest* tmr = sorted_requests[i];
    cout << String::format("** WAIT FOR REQUEST tmr={0}\n",tmr);
  }
#endif

  for (Integer i = 0; i < nb_request; ++i) {
    SharedMemoryMessageRequest* tmr = sorted_requests[i];
    if (tmr) {
      SubQueue* sub_queue = tmr->queue();
      sub_queue->wait(tmr);
      tmr->destroy();
    }
    if (requests[i].hasSubRequest())
      ARCANE_THROW(NotImplementedException, "handling of sub requests");
    requests[i].reset();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::
waitSome(Int32 rank, ArrayView<Parallel::Request> requests,
         ArrayView<bool> requests_done, bool is_non_blocking)
{
  requests_done.fill(false);
  auto sub_queue = _getSubQueue(MessageRank(rank));
  sub_queue->waitSome(requests, requests_done, is_non_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto SharedMemoryMessageQueue::
addReceive(const PointToPointMessageInfo& message, ReceiveBufferInfo buf) -> Request
{
  auto* sq = _getSourceSubQueue(message);
  return _request(sq->addReceive(_getNextRequestId(), message, buf));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto SharedMemoryMessageQueue::
addSend(const PointToPointMessageInfo& message, SendBufferInfo buf) -> Request
{
  auto* sq = _getDestinationSubQueue(message);
  return _request(sq->addSend(_getNextRequestId(), message, buf));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto SharedMemoryMessageQueue::
probe(const MP::PointToPointMessageInfo& message) -> MessageId
{
  auto* sq = _getSourceSubQueue(message);
  return sq->probe(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto SharedMemoryMessageQueue::
legacyProbe(const MP::PointToPointMessageInfo& message) -> MessageSourceInfo
{
  auto* sq = _getSourceSubQueue(message);
  return sq->legacyProbe(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMessageQueue::
setTraceMng(Int32 rank, ITraceMng* tm)
{
  _getSubQueue(MessageRank(rank))->setTraceMng(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto SharedMemoryMessageQueue::
_request(SharedMemoryMessageRequest* tmr) -> Request
{
  return Request(0, this, tmr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
