// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* ISerializeMessageList.h                                     (C) 2000-2020 */
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

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une liste de messages de sérialisation.
 *
 * Les instances de cette classe sont en général créées via la
 * méthode mpCreateSerializeMessageListRef().
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

  //! Envoie les messages de la liste qui ne l'ont pas encore été
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
   * \brief Créé un message de sérialisation.
   *
   * Le message peut être un message d'envoie ou de réception.
   *
   * Si le message est de réception (MsgReceive), il est possible de
   * spécifier un rang nul pour indiquer qu'on souhaite recevoir de n'importe
   * qui.
   */
  virtual Ref<ISerializeMessage>
  createMessage(MessageRank destination,
                ePointToPointMessageType type) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

