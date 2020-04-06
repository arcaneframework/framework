// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiSerializeMessage.h                                       (C) 2000-2020 */
/*                                                                           */
/* Encapsulation de ISerializeMessage pour MPI.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGE_H
#define ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/serialize/SerializeGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiSerializeMessage
{
 public:
  MpiSerializeMessage(ISerializeMessage* message,Integer index);
  ~MpiSerializeMessage();
 public:
  ISerializeMessage* message() const { return m_message; }
  BasicSerializer* serializeBuffer() const { return m_serialize_buffer; }
  Integer messageNumber() const { return m_message_number; }
  void incrementMessageNumber() { ++m_message_number; }
 private:
  ISerializeMessage* m_message;
  BasicSerializer* m_serialize_buffer;
  //! Index du message
  Integer m_message_index;
  //! Numéro du message recu (0 pour le premier)
  Integer m_message_number;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

