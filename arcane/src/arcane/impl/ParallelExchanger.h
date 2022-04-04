﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelExchanger.h                                         (C) 2000-2022 */
/*                                                                           */
/* Echange d'informations entre processeurs.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLELEXCHANGER_H
#define ARCANE_PARALLELEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"

#include "arcane/IParallelExchanger.h"
#include "arcane/Timer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Echange d'informations entre processeurs.
 * \sa IParallelExchanger
*/
class ARCANE_IMPL_EXPORT ParallelExchanger
: public TraceAccessor
, public IParallelExchanger
{
  friend ARCANE_IMPL_EXPORT Ref<IParallelExchanger> createParallelExchangerImpl(Ref<IParallelMng> pm);

 public:

  [[deprecated("Y2022: Use Arcane::createParallelExchangerImpl() instead")]]
  ParallelExchanger(IParallelMng* pm);
  ~ParallelExchanger() override;

 private:

  ParallelExchanger(Ref<IParallelMng> pm);

 public:

  bool initializeCommunicationsMessages() override;
  void initializeCommunicationsMessages(Int32ConstArrayView recv_ranks) override;
  void processExchange() override;
  void processExchange(const ParallelExchangerOptions& options) override;

 public:

  IParallelMng* parallelMng() const override;
  Integer nbSender() const override { return m_send_ranks.size(); }
  Int32ConstArrayView senderRanks() const override { return m_send_ranks; }
  void addSender(Int32 rank) override { m_send_ranks.add(rank); }
  ISerializeMessage* messageToSend(Integer i) override;
  Integer nbReceiver() const override { return m_recv_ranks.size(); }
  Int32ConstArrayView receiverRanks() override { return m_recv_ranks; }
  ISerializeMessage* messageToReceive(Integer i) override;

  void setExchangeMode(eExchangeMode mode) override { m_exchange_mode = mode; }
  eExchangeMode exchangeMode() const override { return m_exchange_mode; }

  void setVerbosityLevel(Int32 v) override;
  Int32 verbosityLevel() const override { return m_verbosity_level; }

  void setName(const String& name) override;
  String name() const override { return m_name; }

 private:
  
  Ref<IParallelMng> m_parallel_mng;

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
  SerializeMessage* m_own_send_message = nullptr;

  //! Message reçu par soi-même.
  SerializeMessage* m_own_recv_message = nullptr;

  //! Mode d'échange.
  eExchangeMode m_exchange_mode = EM_Independant;

  //! Niveau de verbosité
  Int32 m_verbosity_level = 0;

  //! Nom de l'instance utilisé pour l'affichage
  String m_name;

  //! Timer pour mesurer le temps passé dans les échanges
  Timer m_timer;

 private:

  void _initializeCommunicationsMessages();
  void _processExchangeCollective();
  void _processExchangeWithControl(Int32 max_pending_message);
  void _processExchange(const ParallelExchangerOptions& options);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT Ref<IParallelExchanger>
createParallelExchangerImpl(Ref<IParallelMng> pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
