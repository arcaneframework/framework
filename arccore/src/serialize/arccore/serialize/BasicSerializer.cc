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
/* BasicSerializer.cc                                          (C) 2000-2021 */
/*                                                                           */
/* Implémentation simple de 'ISerializer'.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/BasicSerializer.h"
#include "arccore/serialize/BasicSerializerInternal.h"

#include "arccore/base/ArgumentException.h"
#include "arccore/base/FatalErrorException.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'un buffer de sérialisation contigu en mémoire.
 *
 * Cette implémentation permet de sérialiser les données dans une zone
 * contigue en mémoire et ainsi de l'envoyer en une seule fois
 * via MPI par exemple.
 * Le buffer est composé d'un première partie contenant NB_SIZE_ELEM
 * objets de type Integer. Cette partie contient le nombre d'élément
 * sérialisé de chaque type (Real, Int64, Int32, Int16, Byte). La deuxième
 * partie contient les données sérialisées proprement dites.
 *
 * Lorsque cette classe est utilisée dans le cadre d'un appel MPI,
 * on utilise le tableau \a m_size_copy_buffer pour envoyer les tailles.
 * lors du premier message lorsque le message complet est envoyé en plusieurs fois.
 * La norme MPI indique effectivement qu'un buffer utilisé lors d'un appel MPI ne doit plus être
 * utilisé tant que cet appel n'est pas terminé.
 */
class BasicSerializerNewImpl
: public BasicSerializer::Impl
{
  //! Index du tag pour identifier qu'il s'agit d'une sérialisation
  static constexpr int IDX_TAG = 0;
  //! Tag identifiant la sérialisation
  static constexpr Int64 SERIALIZE_TAG = 0x7a9b3cd0;
  //! Version de la sérialisation
  static constexpr int IDX_VERSION = 1;
  //! Champ réservé pour des informations supplémentaires (par exemple compression)
  static constexpr int IDX_RESERVED1 = 2;

  //! Position du champs indiquant la taille totale de la sérialisation
  static constexpr int IDX_TOTAL_SIZE = 3;

  static constexpr int IDX_NB_BYTE = 4;
  static constexpr int IDX_NB_FLOAT16 = 5;
  static constexpr int IDX_NB_FLOAT32 = 6;
  static constexpr int IDX_NB_FLOAT64 = 7;
  static constexpr int IDX_NB_FLOAT128 = 8;
  static constexpr int IDX_NB_INT16 = 9;
  static constexpr int IDX_NB_INT32 = 10;
  static constexpr int IDX_NB_INT64 = 11;
  static constexpr int IDX_NB_INT128 = 12;

  // Laisse de la place pour de nouveaux types.
  static constexpr int IDX_POS_BYTE = 32;
  static constexpr int IDX_POS_FLOAT16 = 33;
  static constexpr int IDX_POS_FLOAT32 = 34;
  static constexpr int IDX_POS_FLOAT64 = 35;
  static constexpr int IDX_POS_FLOAT128 = 36;
  static constexpr int IDX_POS_INT16 = 37;
  static constexpr int IDX_POS_INT32 = 38;
  static constexpr int IDX_POS_INT64 = 39;
  static constexpr int IDX_POS_INT128 = 40;

  static constexpr Integer NB_SIZE_ELEM = 128;
  // La taille de l'alignement est aussi un diviseur de la mémoire
  // allouée pour le message. Elle ne doit pas être modifiée sans modifier
  // la gestion MPI de la sérialisation.
  static constexpr Integer ALIGN_SIZE = BasicSerializer::paddingSize();

 public:

  //! Tableau contenant les données sérialisées
  UniqueArray<Byte> m_buffer;

  //! Vue alignée sur ALIGN_SIZE du m_buffer
  Span<Byte> m_buffer_view;

  Span<Real> m_real_view; //!< Vue sur les reels;
  Span<Int64> m_int64_view; //!< Vue sur les entiers 64 bits
  Span<Int32> m_int32_view; //!< Vue sur les entiers 32 bits
  Span<Int16> m_int16_view; //!< Vue sur les entiers 16 bits
  Span<Byte> m_byte_view; //!< Vur les octets
  
  Int64ArrayView m_sizes_view; //!< Vue pour les tailles (doit être un multiple de ALIGN_SIZE);

  /*!
   * \brief Copie des tailles utilisée pour l'envoie en plusieurs fois.
   *
   * Seuls les premiers éléments (actuellement 40) sont utilisés mais
   * la taille de cette structure doit être un multiple de ALIGN_SIZE.
   */
  Int64 m_size_copy_buffer[NB_SIZE_ELEM];

 public:
  Span<Real> getRealBuffer() override { return m_real_view; }
  Span<Int16> getInt16Buffer() override { return m_int16_view; }
  Span<Int32> getInt32Buffer() override { return m_int32_view; }
  Span<Int64> getInt64Buffer() override { return m_int64_view; }
  Span<Byte> getByteBuffer() override { return m_byte_view; }
  void allocateBuffer(Int64 nb_real,Int64 nb_int16,Int64 nb_int32,
                      Int64 nb_int64,Int64 nb_byte) override
  {
    Int64 total = getPaddingSize(NB_SIZE_ELEM,sizeof(Int64));

    Int64 real_position = total;
    Int64 padded_real_size = getPaddingSize(nb_real,sizeof(Real));
    total += padded_real_size;

    Int64 int16_position = total;
    Int64 padded_int16_size = getPaddingSize(nb_int16,sizeof(Int16));
    total += padded_int16_size;

    Int64 int32_position = total;
    Int64 padded_int32_size = getPaddingSize(nb_int32,sizeof(Int32));
    total += padded_int32_size;

    Int64 int64_position = total;
    Int64 padded_int64_size = getPaddingSize(nb_int64,sizeof(Int64));
    total += padded_int64_size;

    Int64 byte_position = total;
    Int64 padded_byte_size = getPaddingSize(nb_byte,sizeof(Byte));
    total += padded_byte_size;

    _allocBuffer(total);

    m_sizes_view = Int64ArrayView(NB_SIZE_ELEM,(Int64*)&m_buffer_view[0]);

    m_sizes_view[IDX_TAG] = SERIALIZE_TAG;
    m_sizes_view[IDX_VERSION] = 1;
    m_sizes_view[IDX_RESERVED1] = 0;

    m_sizes_view[IDX_TOTAL_SIZE] = total;
    m_sizes_view[IDX_NB_FLOAT64] = nb_real;
    m_sizes_view[IDX_NB_INT64] = nb_int64;
    m_sizes_view[IDX_NB_INT32] = nb_int32;
    m_sizes_view[IDX_NB_INT16] = nb_int16;
    m_sizes_view[IDX_NB_BYTE] = nb_byte;

    m_sizes_view[IDX_POS_FLOAT64] = real_position;
    m_sizes_view[IDX_POS_INT64] = int64_position;
    m_sizes_view[IDX_POS_INT32] = int32_position;
    m_sizes_view[IDX_POS_INT16] = int16_position;
    m_sizes_view[IDX_POS_BYTE] = byte_position;

    m_real_view = Span<Real>((Real*)&m_buffer_view[real_position],nb_real);
    m_int16_view = Span<Int16>((Int16*)&m_buffer_view[int16_position],nb_int16);
    m_int32_view = Span<Int32>((Int32*)&m_buffer_view[int32_position],nb_int32);
    m_int64_view = Span<Int64>((Int64*)&m_buffer_view[int64_position],nb_int64);
    m_byte_view = Span<Byte>((Byte*)&m_buffer_view[byte_position],nb_byte);

    _checkAlignment();
  }

  void copy(Impl* rhs) override
  {
    m_real_view.copy(rhs->getRealBuffer());
    m_int64_view.copy(rhs->getInt64Buffer());
    m_int32_view.copy(rhs->getInt32Buffer());
    m_int16_view.copy(rhs->getInt16Buffer());
    m_byte_view.copy(rhs->getByteBuffer());

    _checkAlignment();
  }

  Span<Byte> globalBuffer() override
  {
    return m_buffer_view;
  }

  Span<const Byte> globalBuffer() const override
  {
    return m_buffer_view;
  }

  Int64ConstArrayView sizesBuffer() const override
  {
    return m_sizes_view;
  }

  void preallocate(Int64 size) override
  {
    _allocBuffer(size);
    m_sizes_view = Int64ArrayView(NB_SIZE_ELEM,(Int64*)&m_buffer_view[0]);
  }

  void setFromSizes() override
  {
    Int64 tag_id = m_sizes_view[IDX_TAG];
    if (tag_id!=SERIALIZE_TAG)
      ARCCORE_FATAL("Bad tag id '{0}' for serializer (expected={1})."
                    "The data are not from a BasicSerializer. SizeView={2}",
                    tag_id,SERIALIZE_TAG,m_sizes_view);
    Int64 version_id = m_sizes_view[IDX_VERSION];
    if (version_id!=1)
      ARCCORE_FATAL("Bad version '{0}' for serializer. Only version 1 is allowed",version_id);

    Int64 nb_real = m_sizes_view[IDX_NB_FLOAT64];
    Int64 nb_int64 = m_sizes_view[IDX_NB_INT64];
    Int64 nb_int32 = m_sizes_view[IDX_NB_INT32];
    Int64 nb_int16 = m_sizes_view[IDX_NB_INT16];
    Int64 nb_byte = m_sizes_view[IDX_NB_BYTE];

    Int64 real_position = m_sizes_view[IDX_POS_FLOAT64];
    Int64 int64_position = m_sizes_view[IDX_POS_INT64];
    Int64 int32_position = m_sizes_view[IDX_POS_INT32];
    Int64 int16_position = m_sizes_view[IDX_POS_INT16];
    Int64 byte_position = m_sizes_view[IDX_POS_BYTE];

    m_real_view = Span<Real>((Real*)&m_buffer_view[real_position],nb_real);
    m_int16_view = Span<Int16>((Int16*)&m_buffer_view[int16_position],nb_int16);
    m_int32_view = Span<Int32>((Int32*)&m_buffer_view[int32_position],nb_int32);
    m_int64_view = Span<Int64>((Int64*)&m_buffer_view[int64_position],nb_int64);
    m_byte_view = Span<Byte>((Byte*)&m_buffer_view[byte_position],nb_byte);

    _checkAlignment();
  }

  ByteConstArrayView copyAndGetSizesBuffer() override
  {
    // Recopie dans \a m_size_copy_buffer les valeurs de \a m_size_view
    // et retourne un pointeur sur \a m_size_copy_buffer.
    Int64ArrayView copy_buf(NB_SIZE_ELEM,m_size_copy_buffer);
    copy_buf.copy(m_sizes_view);
    ByteConstArrayView bytes(sizeof(m_size_copy_buffer),(const Byte*)m_size_copy_buffer);
    return bytes;
  }

  Int64 totalSize() const override
  {
    return m_sizes_view[IDX_TOTAL_SIZE];
  }

  void printSizes(std::ostream& o) const override
  {
    Int64ConstArrayView sbuf_sizes = this->sizesBuffer();
    Int64 total_size = totalSize();
    Span<Byte> bytes = m_buffer_view;
    o << " bytes " << bytes.size()
      << " total_size " << total_size
      << " float64 " << sbuf_sizes[IDX_NB_FLOAT64]
      << " int64 " << sbuf_sizes[IDX_NB_INT64]
      << " int32 " << sbuf_sizes[IDX_NB_INT32]
      << " int16 " << sbuf_sizes[IDX_NB_INT16]
      << " byte " << sbuf_sizes[IDX_NB_BYTE]
      << " ptr=" << (void*)bytes.data();
  }

 protected:
  
  Int64 getPaddingSize(Int64 nb_elem,Int64 elem_size)
  {
    if (nb_elem<0)
      ARCCORE_FATAL("Bad number of element '{0}' (should be >=0)",nb_elem);
    if (elem_size<=0)
      ARCCORE_FATAL("Bad elem_size '{0}'",elem_size);
    Int64 s = nb_elem * elem_size;
    Int64 pad = s % ALIGN_SIZE;
    if (pad==0)
      pad = ALIGN_SIZE;
    Int64 new_size = s + (ALIGN_SIZE-pad);
    if ( (new_size%ALIGN_SIZE)!=0 )
      ARCCORE_FATAL("Bad padding {0}",new_size);
    //std::cout << " nb_elem=" << nb_elem << " elem_size=" << elem_size << " s=" << s << " new_size=" << new_size << '\n';
    return new_size;
  }

  void _checkAlignment()
  {
    _checkAddr(m_real_view.data());
    _checkAddr(m_int16_view.data());
    _checkAddr(m_int32_view.data());
    _checkAddr(m_int64_view.data());
    _checkAddr(m_byte_view.data());
  }

  void _checkAddr(void* ptr)
  {
    Int64 addr = (Int64)ptr;
    if ((addr%ALIGN_SIZE)!=0){
      _printAlignment();
      ARCCORE_FATAL("Bad alignment addr={0} - {1}",addr,(addr % ALIGN_SIZE));
    }
  }

  void _printAlignment()
  {
    for( Integer i=0, n=m_sizes_view.size(); i<n; ++i )
      std::cout << " Size i=" << i << " v=" << m_sizes_view[i] << " pad=" << (m_sizes_view[i] % ALIGN_SIZE) << '\n';
    _printAddr(m_buffer_view.data(),"Buffer");
    _printAddr(m_real_view.data(),"Real");
    _printAddr(m_int16_view.data(),"Int16");
    _printAddr(m_int32_view.data(),"Int32");
    _printAddr(m_int64_view.data(),"Int64");
    _printAddr(m_byte_view.data(),"Byte");
  }

  void _printAddr(void* ptr,const String& name)
  {
    Int64 addr = (Int64)ptr;
    std::cout << "Align type=" << name << " addr=" << addr << " offset=" << (addr % ALIGN_SIZE) << '\n';
  }

  void _allocBuffer(Int64 size)
  {
    if (size<1024)
      size = 1024;
    m_buffer.resize(size+ALIGN_SIZE*4);
    Int64 addr = (Int64)(&m_buffer[0]);
    Int64 padding = addr % ALIGN_SIZE;
    Int64 position = 0;
    if (padding!=0){
      position = ALIGN_SIZE - padding;
    }
    // La taille doit être un multiple de ALIGN_SIZE;
    Int64 new_size = (size + ALIGN_SIZE) - (size%ALIGN_SIZE);
    m_buffer_view = m_buffer.span().subspan(position,new_size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::Impl2::
Impl2()
: m_mode(ModeReserve)
, m_read_mode(ReadReplace)
, m_p(new BasicSerializerNewImpl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::Impl2::
~Impl2()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
reserve(eDataType dt,Int64 n,Int64 nb_put)
{
  ARCCORE_ASSERT((m_mode==ModeReserve),("Bad mode"));
  switch(dt){
  case DT_Real: m_real.m_reserved_size += n; break;
  case DT_Int64: m_int64.m_reserved_size += n; break;
  case DT_Int32: m_int32.m_reserved_size += n; break;
  case DT_Int16: m_int16.m_reserved_size += n; break;
  case DT_Byte: m_byte.m_reserved_size += n; break;
  default:
    ARCCORE_THROW(ArgumentException,"bad datatype v={0}",(int)dt);
  }
  if (m_is_serialize_typeinfo)
    // Pour le type de la donnée.
    m_byte.m_reserved_size += nb_put;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
putType(eBasicDataType t)
{
  if (m_is_serialize_typeinfo){
    Byte b = static_cast<Byte>(t);
    m_byte.put(Span<const Byte>(&b, 1));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
getAndCheckType(eBasicDataType expected_type)
{
  if (!m_is_serialize_typeinfo)
    return;
  Byte b = 0;
  m_byte.get(Span<Byte>(&b,1));
  eBasicDataType t = static_cast<eBasicDataType>(b);
  if (t!=expected_type)
    ARCCORE_FATAL("Bad serialized type t='{0}' int={1}' expected='{2}'",t,(int)t,expected_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
allocateBuffer()
{
  m_p->allocateBuffer(m_real.m_reserved_size,m_int16.m_reserved_size,m_int32.m_reserved_size,
                      m_int64.m_reserved_size,m_byte.m_reserved_size);
  _setViews();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
setFromSizes()
{
  m_p->setFromSizes();
  _setViews();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
_setViews()
{
  m_real.m_buffer = m_p->getRealBuffer();
  m_real.m_current_position = 0;

  m_int64.m_buffer = m_p->getInt64Buffer();
  m_int64.m_current_position = 0;

  m_int32.m_buffer = m_p->getInt32Buffer();
  m_int32.m_current_position = 0;

  m_int16.m_buffer = m_p->getInt16Buffer();
  m_int16.m_current_position = 0;

  m_byte.m_buffer = m_p->getByteBuffer();
  m_byte.m_current_position = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
allocateBuffer(Int64 nb_real,Int64 nb_int16,Int64 nb_int32,
               Int64 nb_int64,Int64 nb_byte)
{
  m_real.m_reserved_size = nb_real;
  m_int64.m_reserved_size = nb_int64;
  m_int32.m_reserved_size = nb_int32;
  m_int16.m_reserved_size = nb_int16;
  m_byte.m_reserved_size = nb_byte;
  allocateBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
copy(const BasicSerializer& rhs)
{
  auto rhs_p = rhs._p();
  Span<Real> real_b = rhs_p->getRealBuffer();
  Span<Int64> int64_b = rhs_p->getInt64Buffer();
  Span<Int32> int32_b = rhs_p->getInt32Buffer();
  Span<Int16> int16_b = rhs_p->getInt16Buffer();
  Span<Byte> byte_b = rhs_p->getByteBuffer();
  allocateBuffer(real_b.size(),int16_b.size(),int32_b.size(),int64_b.size(),byte_b.size());
  m_p->copy(rhs_p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
setMode(eMode new_mode)
{
  if (new_mode==BasicSerializer::ModeGet && new_mode!=m_mode){
    m_real.m_current_position = 0;
    m_int64.m_current_position = 0;
    m_int32.m_current_position = 0;
    m_int16.m_current_position = 0;
    m_byte.m_current_position = 0;
  }

  m_mode = new_mode;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::
BasicSerializer()
: m_p2(new Impl2())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::
BasicSerializer(const BasicSerializer& sb)
: m_p2(new Impl2())
{
  copy(sb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::
~BasicSerializer()
{
  delete m_p2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::Impl* BasicSerializer::
_p() const
{
  return m_p2->m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre ces méthodes privées.
Span<Real> BasicSerializer::realBuffer() { return m_p2->m_real.m_buffer; }
Span<Int64> BasicSerializer::int64Buffer() { return m_p2->m_int64.m_buffer; }
Span<Int32> BasicSerializer::int32Buffer() { return m_p2->m_int32.m_buffer; }
Span<Int16> BasicSerializer::int16Buffer() { return m_p2->m_int16.m_buffer; }
Span<Byte> BasicSerializer::byteBuffer() { return m_p2->m_byte.m_buffer; }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserveSpan(eDataType dt, Int64 n)
{
  m_p2->reserve(dt,n,1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserve(eDataType dt,Int64 n)
{
  m_p2->reserve(dt,n,n);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserve(const String& str)
{
  reserve(DT_Int64,1);
  reserveSpan(str.bytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserveArray(Span<const Real> values)
{
  reserve(DT_Int64,1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int16> values)
{
  reserve(DT_Int64,1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int32> values)
{
  reserve(DT_Int64,1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int64> values)
{
  reserve(DT_Int64,1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Byte> values)
{
  reserve(DT_Int64,1);
  reserveSpan(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Real> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModePut),("Bad mode"));
  m_p2->putType(eBasicDataType::Float64);
  m_p2->m_real.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Int64> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModePut),("Bad mode"));
  m_p2->putType(eBasicDataType::Int64);
  m_p2->m_int64.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Int32> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModePut),("Bad mode"));
  m_p2->putType(eBasicDataType::Int32);
  m_p2->m_int32.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Int16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModePut),("Bad mode"));
  m_p2->putType(eBasicDataType::Int16);
  m_p2->m_int16.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Byte> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModePut),("Bad mode"));
  m_p2->putType(eBasicDataType::Byte);
  m_p2->m_byte.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(const String& str)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModePut),("Bad mode"));
  Int64 len = str.length();
  putInt64(len);
  putSpan(str.bytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
putArray(Span<const Real> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Int16> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Int32> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Int64> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Byte> values)
{
  putInt64(values.size());
  putSpan(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Real> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModeGet),("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Float64);
  m_p2->m_real.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int64> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModeGet),("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int64);
  m_p2->m_int64.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int32> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModeGet),("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int32);
  m_p2->m_int32.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModeGet),("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int16);
  m_p2->m_int16.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Byte> values)
{
  ARCCORE_ASSERT((m_p2->m_mode==ModeGet),("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Byte);
  m_p2->m_byte.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getArray(Array<Real>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Int16>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Int32>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Int64>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Byte>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
get(String& str)
{
  // TODO: il faudrait utiliser des Int64 mais cela casse la compatibilité.
  // A étudier.
  ARCCORE_ASSERT((m_p2->m_mode==ModeGet),("Bad mode"));
  Int64 len = getInt64();
  UniqueArray<Byte> bytes(len);
  getSpan(bytes);
  str = String(bytes.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
allocateBuffer()
{
  m_p2->allocateBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::eMode BasicSerializer::
mode() const
{
  return m_p2->m_mode;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
setMode(eMode new_mode)
{
  m_p2->setMode(new_mode);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer::eReadMode BasicSerializer::
readMode() const
{
  return m_p2->m_read_mode;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
setReadMode(eReadMode new_read_mode)
{
  m_p2->m_read_mode = new_read_mode;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
allocateBuffer(Int64 nb_real,Int64 nb_int16,Int64 nb_int32,
               Int64 nb_int64,Int64 nb_byte)
{
  m_p2->allocateBuffer(nb_real,nb_int16,nb_int32,nb_int64,nb_byte);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
copy(const BasicSerializer& rhs)
{
  m_p2->copy(rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
copy(const ISerializer* from)
{
  auto sbuf = dynamic_cast<const BasicSerializer*>(from);
  if (!sbuf)
    ARCCORE_FATAL("Can only copy from 'BasicSerializer'");
  copy(*sbuf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<Byte> BasicSerializer::
globalBuffer()
{
  return _p()->globalBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const Byte> BasicSerializer::
globalBuffer() const
{
  return _p()->globalBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ByteConstArrayView BasicSerializer::
copyAndGetSizesBuffer()
{
  return _p()->copyAndGetSizesBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView BasicSerializer::
sizesBuffer()
{
  return _p()->sizesBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
preallocate(Int64 size)
{
  return _p()->preallocate(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
setFromSizes()
{
  m_p2->setFromSizes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 BasicSerializer::
totalSize() const
{
  return _p()->totalSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
printSizes(std::ostream& o) const
{
  _p()->printSizes(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
initFromBuffer(Span<const Byte> buf)
{
  Int64 nb_byte = buf.size();
  setMode(ISerializer::ModeGet);
  preallocate(nb_byte);
  globalBuffer().copy(buf);
  setFromSizes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
setSerializeTypeInfo(bool v)
{
  m_p2->setSerializeTypeInfo(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool BasicSerializer::
isSerializeTypeInfo() const
{
  return m_p2->isSerializeTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
