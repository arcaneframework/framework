// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelExchanger.h                                         (C) 2000-2012 */
/*                                                                           */
/* Echange d'informations entre processeurs.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLELEXCHANGER_H
#define ARCANE_PARALLELEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IParallelExchanger.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Echange d'informations entre processeurs.
*/
class ARCANE_IMPL_EXPORT ParallelExchanger
: public TraceAccessor
, public IParallelExchanger
{
 public:

  ParallelExchanger(IParallelMng* pm);
  virtual ~ParallelExchanger();

 public:

  virtual bool initializeCommunicationsMessages();
  virtual void initializeCommunicationsMessages(Int32ConstArrayView recv_ranks);
  virtual void processExchange();

 public:

  virtual IParallelMng* parallelMng() const { return m_parallel_mng; }
  virtual Integer nbSender() const { return m_send_ranks.size(); }
  virtual Int32ConstArrayView senderRanks() const { return m_send_ranks; }
  virtual void addSender(Int32 rank) { m_send_ranks.add(rank); }
  virtual ISerializeMessage* messageToSend(Integer i);
  virtual Integer nbReceiver() const { return m_recv_ranks.size(); }
  virtual Int32ConstArrayView receiverRanks() { return m_recv_ranks; }
  virtual ISerializeMessage* messageToReceive(Integer i);

  virtual void setExchangeMode(eExchangeMode mode) { m_exchange_mode = mode; }
  virtual eExchangeMode exchangeMode() const { return m_exchange_mode; }

 private:
  
  IParallelMng* m_parallel_mng;

  //! Liste des sous-domaines à envoyer
  Int32UniqueArray m_send_ranks;

  //! Liste des sous-domaines à recevoir
  Int32UniqueArray m_recv_ranks;
  
  //! Liste des message à envoyer et recevoir
  UniqueArray<ISerializeMessage*> m_comms_buf;

  //! Liste des message à recevoir
  UniqueArray<SerializeMessage*> m_recv_serialize_infos;

  //! Liste des message à recevoir
  UniqueArray<SerializeMessage*> m_send_serialize_infos;

  //! Message envoyé à soi-même.
  SerializeMessage* m_own_send_message;

  //! Message reçu par soi-même.
  SerializeMessage* m_own_recv_message;

  //! Mode d'échange.
  eExchangeMode m_exchange_mode;

 private:

  void _initializeCommunicationsMessages();
  void _processExchangeCollective();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

