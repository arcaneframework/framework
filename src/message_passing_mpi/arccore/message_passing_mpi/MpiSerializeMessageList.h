// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiSerializeMessageList.h                                   (C) 2000-2020 */
/*                                                                           */
/* Implémentation de ISerializeMessageList pour MPI.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGELIST_H
#define ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGELIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/ISerializeMessageList.h"
#include "arccore/message_passing/Request.h"
#include "arccore/trace/TraceGlobal.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/serialize/SerializeGlobal.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MyMpiParallelMng;
class MpiSerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiSerializeMessageRequest
{
 public:
  MpiSerializeMessageRequest()
  : m_mpi_message(0), m_request() {}
  MpiSerializeMessageRequest(MpiSerializeMessage* mpi_message,Request request)
  : m_mpi_message(mpi_message), m_request(request) {}
 public:
  MpiSerializeMessage* m_mpi_message;
  Request m_request;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation MPI de la gestion des 'ISerializeMessage'.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiSerializeMessageList
: public ISerializeMessageList
{
 private:

  class _SortMessages;

 public:

  MpiSerializeMessageList(MpiSerializeDispatcher* dispatcher);

 public:

  void addMessage(ISerializeMessage* msg) override;
  void processPendingMessages() override;
  Integer waitMessages(eWaitType wait_type) override;
  Ref<ISerializeMessage> createMessage(MessageRank source,MessageRank destination,
                                       ePointToPointMessageType type) override;

  Request _processOneMessageGlobalBuffer(MpiSerializeMessage* msm,int source,int mpi_tag);
  Request _processOneMessage(MpiSerializeMessage* msm, int source, int mpi_tag);

 private:

  Integer _waitMessages(eWaitType wait_type);
  Integer _waitMessages2(eWaitType wait_type);

 private:

  MpiSerializeDispatcher* m_dispatcher = nullptr;
  MpiAdapter* m_adapter = nullptr;
  ITraceMng* m_trace = nullptr;
  UniqueArray<MpiSerializeMessage*> m_messages_to_process;
  UniqueArray<MpiSerializeMessageRequest> m_messages_request;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End Namespace  Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

