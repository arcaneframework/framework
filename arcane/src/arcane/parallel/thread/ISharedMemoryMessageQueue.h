// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISharedMemoryMessageQueue.h                                 (C) 2000-2023 */
/*                                                                           */
/* Interface of a shared memory message queue.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_ISHAREDMEMORYMESSAGEQUEUE_H
#define ARCANE_PARALLEL_THREAD_ISHAREDMEMORYMESSAGEQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"

#include "arcane/parallel/thread/ArcaneThread.h"
#include "arcane/Parallel.h"

#include "arcane/ArcaneTypes.h"

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
using ByteSpan = Arccore::ByteSpan;
using ByteConstSpan = Arccore::ByteConstSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Send buffer information.
 *
 * Contains either a ByteConstSpan or an ISerializer.
 */
class ARCANE_THREAD_EXPORT SendBufferInfo
{
 public:
  SendBufferInfo() = default;
  SendBufferInfo(ConstMemoryView memory_buffer)
  : m_memory_buffer(memory_buffer){}
  SendBufferInfo(const ISerializer* serializer)
  : m_serializer(serializer){}
 public:
  ByteConstSpan memoryBuffer() { return m_memory_buffer.bytes(); }
  const ISerializer* serializer() { return m_serializer; }
 private:

  ConstMemoryView m_memory_buffer;
  const ISerializer* m_serializer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Receive buffer information.
 *
 * Contains either a ByteSpan or an ISerializer.
 */
class ARCANE_THREAD_EXPORT ReceiveBufferInfo
{
 public:
  ReceiveBufferInfo() = default;
  explicit ReceiveBufferInfo(MutableMemoryView memory_buffer)
  : m_memory_buffer(memory_buffer){}
  explicit ReceiveBufferInfo(ISerializer* serializer)
  : m_serializer(serializer){}
 public:
  ByteSpan memoryBuffer() { return m_memory_buffer.bytes(); }
  ISerializer* serializer() { return m_serializer; }
 private:
  MutableMemoryView m_memory_buffer;
  ISerializer* m_serializer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a message queue with threads.
 */
class ARCANE_THREAD_EXPORT ISharedMemoryMessageQueue
: public IRequestCreator
{
 public:

  using MessageSourceInfo = Parallel::MessageSourceInfo;

 public:

  virtual ~ISharedMemoryMessageQueue() = default;

 public:

  virtual void init(Integer nb_thread) =0;
  virtual void waitAll(ArrayView<Request> requests) =0;
  virtual void waitSome(Int32 rank,ArrayView<Request> requests,
                        ArrayView<bool> requests_done,bool is_non_blocking) =0;
  virtual void setTraceMng(Int32 rank,ITraceMng* tm) =0;

 public:

  virtual Request addReceive(const PointToPointMessageInfo& message,ReceiveBufferInfo buf) =0;
  virtual Request addSend(const PointToPointMessageInfo& message,SendBufferInfo buf) =0;
  virtual MessageId probe(const PointToPointMessageInfo& message) =0;
  virtual MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
