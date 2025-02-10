// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializeMessage.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface d'un message de sérialisation entre sous-domaines.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGE_H
#define ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/serialize/SerializeGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un message de sérialisation entre IMessagePassingMng.
 *
 * Un message de sérialisation consiste en une série d'octets envoyés
 * d'un rang source() à un rang destination().
 * Si isSend() est vrai, c'est source() qui envoie à destination(),
 * sinon c'est l'inverse.
 * S'il s'agit d'un message de réception, le serializer() est alloué
 * et remplit automatiquement. Pour que le parallélisme fonctionne correctement,
 * il faut qu'à un message d'envoi corresponde un message de réception
 * complémentaire (envoyé par le rang destination()).
 *
 * Le message peut-être non bloquant. Un message peut-être détruit
 * lorsque sa propriété finished() est vrai.
 */
class ARCCORE_MESSAGEPASSING_EXPORT ISerializeMessage
{
 public:

  enum eMessageType
  {
    MT_Send,
    MT_Recv,
    MT_Broadcast
  };
 //! Stratégie d'envoi/réception
  enum class eStrategy
  {
    //! Stratégie par défaut.
    Default,
    /*!
     * \brief Stratégie utilisant un seul message si possible.
     *
     * Cela suppose d'utiliser la fonction mpProbe()
     * pour connaitre la taille du message
     * avant de poster la réception.
     */
    OneMessage
  };

  virtual ~ISerializeMessage() = default; //!< Libère les ressources.

 public:

  //! \a true s'il faut envoyer, \a false s'il faut recevoir
  virtual bool isSend() const =0;

  //! Type du message
  virtual eMessageType messageType() const =0;

  /*!
   * \brief Rang du destinataire (si isSend() est vrai) ou envoyeur.
   *
   * Dans le cas d'une réception, il est possible de spécifier n'importe quel
   * rang en spécifiant A_NULL_RANK.
   */
  ARCCORE_DEPRECATED_2020("Use destination() instead")
  virtual Int32 destRank() const =0;

  /*!
   * \brief Rang du destinataire (si isSend() est vrai) ou de l'envoyeur.
   *
   * Dans le cas d'une réception, le rang peut valoir nul pour indiquer
   * qu'on souhaite recevoir de n'importe qui.
   * rang en spécifiant A_NULL_RANK.
   */
  virtual MessageRank destination() const =0;

  /*!
   * \brief Rang de l'envoyeur du message
   * Voir aussi destRank() pour une interprétation suivant la valeur de isSend()
   */
  ARCCORE_DEPRECATED_2020("Use source() instead")
  virtual Int32 origRank() const =0;

  /*!
   * \brief Rang de l'envoyeur du message
   *
   * Voir aussi destination() pour une interprétation suivant la valeur de isSend()
   */
  virtual MessageRank source() const =0;

  //! Sérialiseur
  virtual ISerializer* serializer() =0;

  //! \a true si le message est terminé
  virtual bool finished() const =0;

  /*!
   * \internal
   * \brief Positionne l'état 'fini' du message.
   */
  virtual void setFinished(bool v) =0;

  /*!
   * \internal
   */
  ARCCORE_DEPRECATED_2020("Use setInternalTag() instead")
  virtual void setTag(Int32 tag) =0;

  /*!
   * \internal
   * \brief Positionne un tag interne pour le message.
   *
   * Ce tag est utile s'il faut envoyer/recevoir plusieurs messages
   * à un même couple origin/destination.
   *
   * Cette méthode est interne à %Arccore.
   */
  virtual void setInternalTag(MessageTag tag) =0;

  /*!
   * \internal
   * \brief Tag interne du message.
   */
  ARCCORE_DEPRECATED_2020("Use internalTag() instead")
  virtual Int32 tag() const =0;

  /*!
   * \internal
   * \brief Tag interne du message.
   */
  virtual MessageTag internalTag() const =0;

  /*!
   * \internal
   * \brief Identificant du message
   */
  virtual MessageId _internalMessageId() const =0;

  /*!
   * \brief Positionne la stratégie d'envoi/réception.
   *
   * La stratégie utilisée doit être la même pour le message envoyé
   * et le message de réception sinon le comportement est indéfini.
   */
  virtual void setStrategy(eStrategy strategy) =0;

  //! Stratégie utilisée pour les envois/réceptions
  virtual eStrategy strategy() const =0;

  /*!
   * \brief Indique si le message a déjà été traité.
   *
   * Si le message a déjà été traité, il n'est pas possible de changer certaines
   * caractéristiques (comme la stratégie ou le tag)
   */
  virtual bool isProcessed() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

