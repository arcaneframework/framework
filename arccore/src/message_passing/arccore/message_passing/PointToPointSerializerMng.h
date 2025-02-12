// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointToPointSerializerMng.h                                 (C) 2000-2025 */
/*                                                                           */
/* Communications point à point par des 'ISerializer'.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_POINTOTPOINTSERIALIZERMNG_H
#define ARCCORE_MESSAGEPASSING_POINTOTPOINTSERIALIZERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/ISerializeMessage.h"
#include "arccore/base/RefDeclarations.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Communications point à point par des 'ISerializer'.
 */
class ARCCORE_MESSAGEPASSING_EXPORT PointToPointSerializerMng
{
  class Impl;

 public:

  PointToPointSerializerMng(IMessagePassingMng* mpm);
  ~PointToPointSerializerMng();

 public:
  PointToPointSerializerMng(const PointToPointSerializerMng&) = delete;
  PointToPointSerializerMng& operator=(const PointToPointSerializerMng&) = delete;
 public:

  //! Gestionnaire de message associé
  IMessagePassingMng* messagePassingMng() const;

  /*!
   * \brief Envoie les messages de la liste qui ne l'ont pas encore été.
   *
   * Il n'est en général pas nécessaire d'appeler cette
   * méthode car cele est fait automatiquement lors de l'appel à waitMessages().
   */
  void processPendingMessages();

  /*!
   * \brief Attend que les messages aient terminé leur exécution.
   * 
   * Le type d'attente est spécifié par \a wt.
   *
   * \return le nombre de messages complètement exécutés ou (-1) s'ils
   * l'ont tous été.
   */
  Integer waitMessages(eWaitType wt,std::function<void(ISerializeMessage*)> functor);

  //! Indique s'il reste des messages qui ne sont pas encore terminés.
  bool hasMessages() const;

  /*!
   * \brief Créé un message de sérialisation en réception
   *
   * \a sender_rank est le rang de celui qui envoie le message correspondant.
   * Il est possible de spécifier un rang nul pour indiquer qu'on
   * souhaite recevoir de n'importe qui.
   */
  Ref<ISerializeMessage> addReceiveMessage(MessageRank sender_rank);

  /*!
   * \brief Créé un message de sérialisation en réception
   *
   * \a sender_rank est le rang de celui qui envoie le message correspondant.
   * Il est possible de spécifier un rang nul pour indiquer qu'on
   * souhaite recevoir de n'importe qui.
   */
  Ref<ISerializeMessage> addReceiveMessage(MessageId message_id);

  /*!
   * \brief Créé message de sérialisation en envoi.
   */
  Ref<ISerializeMessage> addSendMessage(MessageRank receiver_rank);

  /*!
   * \brief Tag par défaut utilisé pour les messages.
   *
   * Cette méthode ne peut être appelée que s'il n'y a pas de messages
   * en cours (hasMessages()==false). Tous les rangs de messagePassingMng()
   * doivent utiliser le même tag.
   */
  void setDefaultTag(MessageTag default_tag);

  /*!
   * \brief Stratégie utilisée pour les messages.
   *
   * Cette méthode ne peut être appelée que s'il n'y a pas de messages
   * en cours (hasMessages()==false). Tous les rangs de messagePassingMng()
   * doivent utiliser la même stratégie.
   */
  void setStrategy(ISerializeMessage::eStrategy strategy);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

