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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Message utilisant un BasicSerializer.
 *
 * Un message consiste en une série d'octets envoyés d'un rang
 * (origRank()) à un autre (destRank()). Si isSend() est vrai,
 * c'est origRank() qui envoie à destRank(), sinon c'est l'inverse.
 * S'il s'agit d'un message de réception, le serializer() est alloué
 * et remplit automatiquement.
 *
 * Pour que le parallélisme fonctionne correctement, il faut qu'un message
 * complémentaire à celui-ci soit envoyé par destRank().
 */
class ARCCORE_MESSAGEPASSING_EXPORT BasicSerializeMessage
: public ISerializeMessage
{
 public:
  BasicSerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype);
  ~BasicSerializeMessage();
  BasicSerializeMessage& operator=(const BasicSerializeMessage&) = delete;
  BasicSerializeMessage(const BasicSerializeMessage&) = delete;
 protected:
  BasicSerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype,
                        BasicSerializer* serializer);
 public:
  bool isSend() const override { return m_is_send; }
  eMessageType messageType() const override { return m_message_type; }
  Int32 destRank() const override { return m_dest_rank; }
  Int32 origRank() const override { return m_orig_rank; }
  ISerializer* serializer() override { return m_buffer; }
  BasicSerializer& buffer() { return *m_buffer; }
  bool finished() const override { return m_finished; }
  void setFinished(bool v) override { m_finished = v; }
  void setTag(Int32 tag) override { m_tag = tag; }
  Int32 tag() const override { return m_tag; }
 private:
  Int32 m_orig_rank; //!< Rang de l'expéditeur de la requête
  Int32 m_dest_rank; //!< Rang du destinataire du message
  Int32 m_tag;
  eMessageType m_message_type; //!< Type du message
  bool m_is_send; //!< \c true si envoie, \c false si réception
  BasicSerializer* m_buffer = nullptr; //!< Tampon contenant les infos
  bool m_finished; //!< \c true si message terminé
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

