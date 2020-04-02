// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* BasicSerialize.h                                            (C) 2000-2020 */
/*                                                                           */
/* Message utilisant un BasicSerializer.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_BASICSERIALIZEMESSAGE_H
#define ARCCORE_MESSAGEPASSING_BASICSERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/BasicSerializer.h"
#include "arccore/message_passing/ISerializeMessage.h"
#include "arccore/message_passing/MessageRank.h"
#include "arccore/message_passing/MessageTag.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Message de sérialisation utilisant un BasicSerializer.
 *
 * Cette classe est interne à %Arccore et ne doit pas être utilisée
 * directement. Si on souhaite créer une instance de ISerializeMessage,
 * il faut passer créer une liste de messages via
 * ISerializeMessageList::createMessage();
 *
 * Un message consiste en une série d'octets envoyés ou recu d'un rang
 * (source()) à un autre (destination()). Si isSend() est vrai,
 * c'est source() qui envoie des octets à destination(), sinon c'est source()
 * qui recoit des octets de destination().
 * S'il s'agit d'un message de réception, le serializer() est alloué
 * et remplit automatiquement.
 */
class ARCCORE_MESSAGEPASSING_EXPORT BasicSerializeMessage
: public ISerializeMessage
{
 public:
  ARCCORE_DEPRECATED_2020("Use ISerializeMessageList::create() static method instead")
  BasicSerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype);
 protected:
  BasicSerializeMessage(MessageRank orig_rank,MessageRank dest_rank,ePointToPointMessageType mtype);
 public:
  ~BasicSerializeMessage();
  BasicSerializeMessage& operator=(const BasicSerializeMessage&) = delete;
  BasicSerializeMessage(const BasicSerializeMessage&) = delete;
 protected:
  BasicSerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype,
                        BasicSerializer* serializer);
 public:
  static Ref<ISerializeMessage>
  create(MessageRank source,MessageRank destination,ePointToPointMessageType type);
 public:
  bool isSend() const override { return m_is_send; }
  eMessageType messageType() const override { return m_old_message_type; }
  Int32 destRank() const override { return m_dest_rank.value(); }
  Int32 origRank() const override { return m_orig_rank.value(); }
  MessageRank destination() const override { return m_dest_rank; }
  MessageRank source() const override { return m_orig_rank; }
  ISerializer* serializer() override { return m_buffer; }
  BasicSerializer& buffer() { return *m_buffer; }
  bool finished() const override { return m_finished; }
  void setFinished(bool v) override { m_finished = v; }
  void setTag(Int32 tag) override { m_tag = MessageTag(tag); }
  Int32 tag() const override { return m_tag.value(); }
  void setInternalTag(MessageTag tag) override { m_tag = tag; }
  MessageTag internalTag() const override { return m_tag; }
 private:
  MessageRank m_orig_rank; //!< Rang de l'expéditeur de la requête
  MessageRank m_dest_rank; //!< Rang du destinataire du message
  MessageTag m_tag = MessageTag(0);
  eMessageType m_old_message_type; //!< Type du message (obsolète)
  ePointToPointMessageType m_message_type; //!< Type du message
  bool m_is_send; //!< \c true si envoie, \c false si réception
  BasicSerializer* m_buffer = nullptr; //!< Tampon contenant les infos
  bool m_finished = false; //!< \c true si message terminé
 private:
  void _init();
  static ePointToPointMessageType _toP2PType(eMessageType mtype);
  static eMessageType _toMessageType(ePointToPointMessageType mtype);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

