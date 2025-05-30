﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BlockSizes                                                  (C) 2000-2025 */
/*                                                                           */
/* Size info for block matrices                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/utils/Precomp.h>
#include <alien/arcane_tools/IIndexManager.h>
#include "BlockSizes.h"

#include <alien/utils/Precomp.h>
#include <alien/utils/Trace.h>

#include <arccore/message_passing/ISerializeMessageList.h>
#include <arccore/message_passing/Messages.h>

#include <map>
#include <list>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

namespace ArcaneTools {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockSizes::
BlockSizes()
  : m_is_prepared(false) 
  , m_local_size(0)
  , m_max_size(0)
{
  ;
}

/*---------------------------------------------------------------------------*/

struct BlockSizes::EntrySendRequest
{
  EntrySendRequest(){}
  
  ~EntrySendRequest()
  {}

  Arccore::Ref<Arccore::MessagePassing::ISerializeMessage> m_comm;
  Integer m_count = 0;
};

struct BlockSizes::EntryRecvRequest
{
  EntryRecvRequest()
  {
    ;
  }

  ~EntryRecvRequest()
  {}

  Arccore::Ref<Arccore::MessagePassing::ISerializeMessage> m_comm;
  UniqueArray<Integer> m_ids;
};


void
BlockSizes::
prepare(const IIndexManager& index_mng, ConstArrayView<Integer>  block_sizes)
{
  if(m_is_prepared)
    throw FatalErrorException(A_FUNCINFO,"BlockSizes is already prepared");
  
  m_parallel_mng = index_mng.parallelMng();
   
  const Integer offset = index_mng.minLocalIndex();
  
  const Integer size = block_sizes.size();
  

  m_local_sizes.resize(size);
  m_local_offsets.resize(size+1);

  Integer sum = 0, max_size = 0;
  for(Integer i = 0; i < size; ++i) {
    const Integer block_size = block_sizes[i];
    max_size = (max_size > block_size) ? max_size : block_size;
    const Integer index = i + offset;
    m_sizes[index] = m_local_sizes[i] = block_size;
    m_offsets[index] = m_local_offsets[i] = sum;
    sum += block_size;
  }
  m_local_offsets[size] = sum ;
  m_max_size = max_size;
  m_local_size = sum;
  m_is_prepared = true;
  
  const bool is_parallel = (m_parallel_mng!=nullptr) && (m_parallel_mng->commSize()>1);
  if(not is_parallel) return;
  
  const Integer rank = m_parallel_mng->commRank();
 
  typedef std::map<String,EntrySendRequest> SendRequestByEntry;
  typedef std::map<Integer,SendRequestByEntry> SendRequests;
  SendRequests sendRequests;

  typedef SharedArray<Integer> Owners;
  typedef std::map<String,Owners> OwnersByEntry;
  OwnersByEntry owners_by_entry;

  for(IIndexManager::EntryEnumerator e = index_mng.enumerateEntry(); e.hasNext(); ++e) {
    //ConstArrayView<Integer>  all_indexes = e->getAllIndexes();
    ConstArrayView<Integer>  all_lids = e->getAllLocalIds();
    Owners owners = e->getFamily().owners(all_lids);
    const String name = e->getName();
    owners_by_entry[name] = owners;
    for(Integer i = 0; i < owners.size(); ++i) {
      if(owners[i] != rank) sendRequests[owners[i]][name].m_count++;
    }
  }
  
  //Arcane::ISerializeMessageList* messageList = m_parallel_mng->createSerializeMessageList();
  Alien::Ref<ISerializeMessageList> messageList =
      Arccore::MessagePassing::mpCreateSerializeMessageListRef(m_parallel_mng);

  UniqueArray<Integer> sendToDomains(2*m_parallel_mng->commSize(),0);

  for(SendRequests::iterator i = sendRequests.begin(); i != sendRequests.end(); ++i) {
    const Integer destDomainId = i->first;
    SendRequestByEntry& requests = i->second;
    for(SendRequestByEntry::iterator j = requests.begin(); j != requests.end(); ++j) {
      EntrySendRequest& request = j->second;
      const String& nameString = j->first;
      sendToDomains[2*destDomainId + 0] += 1;
      sendToDomains[2*destDomainId + 1] += request.m_count;
      //request.comm = new Arcane::SerializeMessage(m_parallel_mng->commRank(),destDomainId,Arcane::ISerializeMessage::MT_Send);
      //messageList->addMessage(request.comm);
      //Arcane::SerializeBuffer& sbuf = request.comm->buffer();
      request.m_comm = messageList->createAndAddMessage(MessageRank(destDomainId),
                                                        Arccore::MessagePassing::ePointToPointMessageType::MsgSend);
      auto sbuf = request.m_comm->serializer();

      sbuf->setMode(Alien::ISerializer::ModeReserve);
      sbuf->reserve(nameString);                     // Chaine de caractère du nom de l'entrée
      sbuf->reserveInteger(1);                       // Nb d'item
      sbuf->reserve(Alien::ISerializer::DT_Int32,request.m_count); // Les indices demandés
      sbuf->allocateBuffer();
      sbuf->setMode(Alien::ISerializer::ModePut);
      sbuf->put(nameString);
      sbuf->put(request.m_count);
    }
  }
  
  for(IIndexManager::EntryEnumerator e = index_mng.enumerateEntry(); e.hasNext(); ++e) { 
    const String name = e->getName();
    Owners& owners = owners_by_entry[name];
    ConstArrayView<Integer>  all_indexes = e->getAllIndexes();
    for(Integer i = 0; i < owners.size(); ++i) {
      if(owners[i] != rank) sendRequests[owners[i]][name].m_comm->serializer()->put(all_indexes[i]);
    }
  }
  
  UniqueArray<Integer> recvFromDomains(2*m_parallel_mng->commSize());
  Arccore::MessagePassing::mpAllToAll(m_parallel_mng,sendToDomains,recvFromDomains,2);
  
  typedef std::list<EntryRecvRequest> RecvRequests;
  RecvRequests recvRequests;

  for(Integer isd=0, nsd=m_parallel_mng->commSize();isd<nsd;++isd) {
    Integer recvCount = recvFromDomains[2*isd+0];
    while(recvCount-- > 0) {
      auto recvMsg = messageList->createAndAddMessage(MessageRank(isd),
                                                      Arccore::MessagePassing::ePointToPointMessageType::MsgReceive);

      recvRequests.push_back(EntryRecvRequest());
      EntryRecvRequest& recvRequest = recvRequests.back();
      recvRequest.m_comm = recvMsg;
    }
  }
  
  messageList->processPendingMessages();
  messageList->waitMessages(Arccore::MessagePassing::WaitAll);
  messageList.reset();

  //messageList = parallel_mng->createSerializeMessageList();
  messageList =
      Arccore::MessagePassing::mpCreateSerializeMessageListRef(m_parallel_mng);

  for(RecvRequests::iterator i = recvRequests.begin(); i != recvRequests.end(); ++i) {
    EntryRecvRequest & recvRequest = *i;
    String nameString;
    Integer uidCount;
    
    { 
      //Arcane::SerializeBuffer& sbuf = recvRequest.comm->buffer();
      //sbuf.setMode(Arcane::ISerializer::ModeGet);
      auto sbuf = recvRequest.m_comm->serializer();
      sbuf->setMode(Alien::ISerializer::ModeGet);

      
      sbuf->get(nameString);
      uidCount = sbuf->getInteger();
      recvRequest.m_ids.resize(uidCount);
      sbuf->get(recvRequest.m_ids);
      ALIEN_ASSERT((uidCount == recvRequest.m_ids.size()),("Inconsistency detected"));
    }
    
    {
      auto dest = recvRequest.m_comm->destination();
      recvRequest.m_comm.reset();
      recvRequest.m_comm = messageList->createAndAddMessage(dest, Arccore::MessagePassing::ePointToPointMessageType::MsgSend);
      
      //Arcane::SerializeBuffer & sbuf = recvRequest.comm->buffer();
      auto sbuf = recvRequest.m_comm->serializer();

      sbuf->setMode(Alien::ISerializer::ModeReserve);
      sbuf->reserve(nameString);      // Chaine de caractère du nom de l'entrée
      sbuf->reserveInteger(1);        // Nb d'item
      sbuf->reserveInteger(uidCount); // Les tailles
      sbuf->allocateBuffer();
      sbuf->setMode(Alien::ISerializer::ModePut);
      sbuf->put(nameString);
      sbuf->put(uidCount);
    }
  }
  
  for(RecvRequests::iterator i = recvRequests.begin(); i != recvRequests.end(); ++i) {
    EntryRecvRequest & recvRequest = *i;
    //Arcane::SerializeBuffer& sbuf = recvRequest.comm->buffer();
    auto sbuf = recvRequest.m_comm->serializer();
    auto& ids = recvRequest.m_ids;
    for(Integer j = 0; j<ids.size(); ++j) {
      sbuf->putInteger(block_sizes[ids[j]-offset]);
    }
  }
  
  //typedef std::list<Arcane::SerializeMessage*> ReturnedRequests;
  typedef std::list<Alien::Ref<Alien::ISerializeMessage>> ReturnedRequests;
  ReturnedRequests returnedRequests;
 
  typedef std::map<Integer, EntrySendRequest*> SubFastReturnMap;
  typedef std::map<String, SubFastReturnMap> FastReturnMap;
  FastReturnMap fastReturnMap;

  for(SendRequests::iterator i = sendRequests.begin(); i != sendRequests.end(); ++i) {
    const Integer destDomainId = i->first;
    SendRequestByEntry & requests = i->second;
    for(SendRequestByEntry::iterator j = requests.begin(); j != requests.end(); ++j) {
      EntrySendRequest & request = j->second;
      const String nameString = j->first;
      request.m_comm.reset();
      auto msg = messageList->createAndAddMessage(MessageRank(destDomainId),
                                                  Arccore::MessagePassing::ePointToPointMessageType::MsgReceive);

      returnedRequests.push_back(msg);
      
      fastReturnMap[nameString][destDomainId] = &request;
    }
  }
  
  messageList->processPendingMessages();
  messageList->waitMessages(Arccore::MessagePassing::WaitAll);
  messageList.reset();
  
  for(ReturnedRequests::iterator i = returnedRequests.begin(); i != returnedRequests.end(); ++i) {
    //Arcane::SerializeMessage * message = *i;
    //const Integer origDomainId = message->destRank();
    //Arcane::SerializeBuffer& sbuf = message->buffer();
    auto& message = *i;
    auto origDomainId = message->destination().value();
    auto sbuf = message->serializer();
    sbuf->setMode(Alien::ISerializer::ModeGet);
    String nameString;
    sbuf->get(nameString);
    ALIEN_ASSERT((fastReturnMap[nameString][origDomainId] != NULL),("Inconsistency detected"));
    EntrySendRequest & request = *fastReturnMap[nameString][origDomainId];
    request.m_comm = *i;
    const Integer idCount = sbuf->getInteger();
    ALIEN_ASSERT((request.m_count == idCount),("Inconsistency detected"));
  }
  
  for(IIndexManager::EntryEnumerator e = index_mng.enumerateEntry(); e.hasNext(); ++e) {
    ConstArrayView<Integer>  all_indexes = e->getAllIndexes();
    Owners owners = owners_by_entry[(*e).getName()];
    for(Integer i = 0; i < owners.size(); ++i) {
      const Integer index = all_indexes[i];
      if (owners[i] != rank) {
        EntrySendRequest& request = sendRequests[owners[i]][e->getName()];
        ALIEN_ASSERT((request.m_count > 0),("Unexpected empty request"));
        --request.m_count;
        const Integer block_size = request.m_comm->serializer()->getInteger();
        m_sizes[index] = block_size;
        m_max_size = (m_max_size > block_size) ? m_max_size : block_size;
        m_offsets[index] = sum;
        sum += block_size;
      }
    }
  }
  
//   for(IIndexManager::EntryEnumerator e = index_mng.enumerateEntry(); e.hasNext(); ++e) {
//     ConstArrayView<Integer>  all_indexes = e->getAllIndexes();
//      for(Integer i = 0; i < all_indexes.size(); ++i) {
//        const Integer index = all_indexes[i];
//        trace->info() << "index=" << index << " size = " << m_sizes[index] << ", offset = " << m_offsets[index];
//      }
//   }
  
  m_is_prepared = true;
}

/*---------------------------------------------------------------------------*/

Integer 
BlockSizes::
size(Integer index) const
{
  ALIEN_ASSERT((m_is_prepared),("BlockSizes is not prepared"));
 
  ValuePerBlock::const_iterator it = m_sizes.find(index);

  if(it == m_sizes.end())
    throw Alien::FatalErrorException(A_FUNCINFO,"index is not registered");

  return it.value();
}

/*---------------------------------------------------------------------------*/

Integer 
BlockSizes::
sizeFromLocalIndex(Integer index) const
{
  //A. Anciaux
  //ToDo pas faire une mapping --> aller vers un tableau pointeur
  ALIEN_ASSERT((m_is_prepared),("BlockSizes is not prepared"));
 
  return m_local_sizes[index];
}

/*---------------------------------------------------------------------------*/

Integer 
BlockSizes::
offset(Integer index) const
{
  //A. Anciaux
  //ToDo pas faire une mapping --> aller vers un tableau pointeur
  ALIEN_ASSERT((m_is_prepared),("BlockSizes is not prepared"));
 
  ValuePerBlock::const_iterator it = m_offsets.find(index);

  if(it == m_offsets.end())
    throw Alien::FatalErrorException(A_FUNCINFO,"index is not registered");
  
  return it.value();
}



/*---------------------------------------------------------------------------*/

Integer 
BlockSizes::
offsetFromLocalIndex(Integer index) const
{
  ALIEN_ASSERT((m_is_prepared),("BlockSizes is not prepared"));
 
  return m_local_offsets[index];
}

/*---------------------------------------------------------------------------*/

Integer 
BlockSizes::
localSize() const
{
  ALIEN_ASSERT((m_is_prepared),("BlockSizes is not prepared"));
  
  return m_local_size;
}

/*---------------------------------------------------------------------------*/

Integer 
BlockSizes::
maxSize() const
{
  ALIEN_ASSERT((m_is_prepared),("BlockSizes is not prepared"));
  
  return m_max_size;
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
BlockSizes::
sizeOfLocalIndex() const
{
  return m_local_sizes;
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
BlockSizes::
offsetOfLocalIndex() const
{
  return m_local_offsets;
}
 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
