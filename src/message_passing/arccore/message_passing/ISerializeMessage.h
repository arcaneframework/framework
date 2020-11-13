// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializeMessage.h                                         (C) 2000-2020 */
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

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un message de sérialisation entre IParallelMng.
 *
 * Un message de sérialisation consiste en une série d'octets envoyés
 * d'un sous-domaine (origRank()) à un autre (destRank()).
 * Si isSend() est vrai, c'est origRank() qui envoie à destRank(),
 * sinon c'est l'inverse.
 * S'il s'agit d'un message de réception, le serializer() est alloué
 * et remplit automatiquement. Pour que le parallélisme fonctionne correctement,
 * il faut qu'à un message d'envoie corresponde un message de réception
 * complémentaire (envoyé par destRank()).
 *
 * Les messages sont envoyés par IParallelMng::processMessages().
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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

