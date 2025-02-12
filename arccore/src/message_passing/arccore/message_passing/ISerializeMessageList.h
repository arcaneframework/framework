// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializeMessageList.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface d'une liste de messages de sérialisation.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGEMESSAGELIST_H
#define ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGEMESSAGELIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une liste de messages de sérialisation.
 *
 * Les instances de cette classe sont en général créées via la
 * méthode mpCreateSerializeMessageListRef().
 *
 * \code
 * using namespace Arccore;
 * IMessagePassingMng* mpm = ...;
 * Ref<ISerializeMessagList> serialize_list(mpCreateSerializeMessageListRef(mpm));
 * // Ajoute deux messages pour recevoir du rang 1 et 3.
 * Ref<ISerializeMessage> message1(serialize_list->createAndAddMessage(MessageRank(1),MsgReceive));
 * Ref<ISerializeMessage> message2(serialize_list->createAndAddMessage(MessageRank(3),MsgReceive));
 *
 * // Attend que les messages soient terminés
 * // En mode WaitSome ou TestSome, il est possible de savoir si
 * // un message est terminé en appelant la méthode ISerializeMessage::finished()
 * serialize_list->wait(WaitAll);
 *
 * // Récupère les données de sérialisation du premier message
 * ISerializer* sb1 = message1.serializer();
 * sb1->setMode(ISerializer::ModeGet);
 * UniqueArray<Int32> int32_array;
 * sb1->getArray(int32_array);
 * ...
 * \endcode
 */
class ARCCORE_MESSAGEPASSING_EXPORT ISerializeMessageList
{
 public:

  virtual ~ISerializeMessageList() {} //!< Libère les ressources.

 public:

  /*!
   * \brief Ajoute un message à la liste.
   *
   * Le message n'est pas posté tant qu'aucun appel à processPendingMessages()
   * n'a été effectué. L'utilisateur garde la propriété du message qui
   * ne doit pas être détruit tant qu'il n'est pas terminé.
   */
  virtual void addMessage(ISerializeMessage* msg) =0;

  /*!
   * \brief Envoie les messages de la liste qui ne l'ont pas encore été.
   *
   * Cette méthode envoie les messages ajoutés via addMessage() qui ne l'ont
   * pas encore été. Il n'est en général pas nécessaire d'appeler cette
   * méthode car cele est fait automatiquement lors de l'appel à waitMessages().
   */
  virtual void processPendingMessages() =0;

  /*!
   * \brief Attend que les messages aient terminé leur exécution.
   * 
   * Le type d'attente est spécifié par \a wt.
   *
   * Il est ensuite possible de tester si un message est terminé
   * via la méthode ISerializeMessage::isFinished(). Cette classe ne garde
   * aucune référence sur les messages terminés qui peuvent donc être
   * détruits par l'utilisateur dès qu'il n'en a plus besoin.
   *
   * \return le nombre de messages complètement exécutés ou (-1) s'ils
   * l'ont tous été.
   */
  virtual Integer waitMessages(eWaitType wt) =0;

  /*!
   * \brief Créé et ajoute un message de sérialisation.
   *
   * Le message peut être un message d'envoie ou de réception.
   *
   * Si le message est de réception (MsgReceive), il est possible de
   * spécifier un rang nul pour indiquer qu'on souhaite recevoir de n'importe
   * qui.
   *
   * Cette méthode appelle addMessage() pour ajouter automatiquement le
   * message créé à la liste des messages. L'instance ne conserve pas de référence
   * au message créé qui ne doit pas être détruit tant qu'il n'est
   * pas terminé.
   */
  virtual Ref<ISerializeMessage>
  createAndAddMessage(MessageRank destination,
                      ePointToPointMessageType type) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

