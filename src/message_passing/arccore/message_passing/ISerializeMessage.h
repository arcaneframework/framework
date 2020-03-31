// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

  virtual ~ISerializeMessage() {} //!< Libère les ressources.

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
  virtual Int32 destRank() const =0;

  /*!
   * \brief Rang de l'envoyeur du message
   * Voir aussi destRank() pour une interprétation suivant la valeur de isSend()
   */
  virtual Int32 origRank() const =0;

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
   * \brief Positionne un tag pour le message. Ce
   * Ce tag est utile s'il faut envoyer/recevoir plusieurs messages
   * à un même couple origin/destination.
   * Cette méthode est interne à Arcane.
   */
  virtual void setTag(Int32 tag) =0;

  /*!
   * \internal
   * \brief Tag du message.
   */
  virtual Int32 tag() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

