// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* SerializeByteConverter.h                                    (C) 2000-2020 */
/*                                                                           */
/* Wrappeur pour envoyer un tableau d'octets d'un sérialiseur.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_SERIALIZEBYTECONVERTER_H
#define ARCCORE_MESSAGEPASSINGMPI_SERIALIZEBYTECONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
//TODO: mettre cela dans un .cc
#include "arccore/serialize/BasicSerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi::internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Wrappeur pour envoyer un tableau d'octets d'un sérialiseur.
 *
 * \a SpanType doit être un 'Byte' ou un 'const Byte'.
 *
 * Comme MPI utilise un 'int' pour le nombre d'éléments d'un message, on ne
 * peut pas dépasser 2^31 octets pas message. Par contre, les versions 3.0+
 * de MPI supportent des messages dont la longueur dépasse 2^31.
 * On utilise donc un type dérivé MPI contenant N octets (avec N donné
 * par SerializeBuffer::paddingSize()) et on indique à MPI que c'est ce type
 * qu'on envoie. Le nombre d'éléments est donc divisé par N ce qui permet
 * de tenir sur 'int' si la taille du message est inférieure à 2^31 * N octets
 * (en février 2019, N=128 soit des messages de 256Go maximum).
 *
 * \note Pour que cela fonctionne, le tableau \a buffer doit avoir une
 * mémoire allouée arrondie au multiple de N supérieur au nombre d'éléments
 * mais normalement cela est garanti par le SerializeBuffer.
 */
template<typename SpanType>
class SerializeByteConverter
{
 public:
  SerializeByteConverter(Span<SpanType> buffer,MPI_Datatype byte_serializer_datatype)
  : m_buffer(buffer), m_datatype(byte_serializer_datatype), m_final_size(-1)
  {
    Int64 size = buffer.size();
    const Int64 align_size = BasicSerializer::paddingSize();
    if ((size%align_size)!=0)
      ARCANE_FATAL("Buffer size '{0}' is not a multiple of '{1}' Invalid size",size,align_size);
    m_final_size = size / align_size;
  }
  SpanType* data() { return m_buffer.data(); }
  Int64 size() const { return m_final_size; }
  Int64 messageSize() const { return m_buffer.size() * sizeof(Byte); }
  Int64 elementSize() const { return BasicSerializer::paddingSize(); }
  MPI_Datatype datatype() const { return m_datatype; }
 private:
  Span<SpanType> m_buffer;
  MPI_Datatype m_datatype;
  Int64 m_final_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi::internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
