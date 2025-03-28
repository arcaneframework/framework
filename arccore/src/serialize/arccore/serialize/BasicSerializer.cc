// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicSerializer.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Implémentation simple de 'ISerializer'.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/BasicSerializer.h"

#include "arccore/base/ArgumentException.h"
#include "arccore/base/FatalErrorException.h"

#include "arccore/serialize/internal/BasicSerializerInternal.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
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
 * La norme MPI indique effectivement qu'un buffer utilisé lors d'un appel
 * MPI ne doit plus être utilisé tant que cet appel n'est pas terminé.
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
  static constexpr int IDX_NB_INT8 = 13;
  static constexpr int IDX_NB_BFLOAT16 = 14;

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
  static constexpr int IDX_POS_INT8 = 41;
  static constexpr int IDX_POS_BFLOAT16 = 42;

  static constexpr Integer NB_SIZE_ELEM = 128;
  // La taille de l'alignement est aussi un diviseur de la mémoire
  // allouée pour le message. Elle ne doit pas être modifiée sans modifier
  // la gestion MPI de la sérialisation.
  static constexpr Integer ALIGN_SIZE = BasicSerializer::paddingSize();

 public:

  //! Informations sur la taille allouée avec et sans padding.
  struct SizeInfo
  {
   public:

    Int64 m_original_size = 0;
    Int64 m_padded_size = 0;
  };

 public:

  //! Tableau contenant les données sérialisées
  UniqueArray<Byte> m_buffer;

  //! Vue alignée sur ALIGN_SIZE du m_buffer
  Span<Byte> m_buffer_view;

  Span<Real> m_real_view; //!< Vue sur les reels;
  Span<Int64> m_int64_view; //!< Vue sur les entiers 64 bits
  Span<Int32> m_int32_view; //!< Vue sur les entiers 32 bits
  Span<Int16> m_int16_view; //!< Vue sur les entiers 16 bits
  Span<Byte> m_byte_view; //!< Vue les octets
  Span<Int8> m_int8_view; //!< Vue les Int8
  Span<Float16> m_float16_view; //!< Vue les Float16
  Span<BFloat16> m_bfloat16_view; //!< Vue les BFloat16
  Span<Float32> m_float32_view; //!< Vue les Float32
  Span<Float128> m_float128_view; //!< Vue les Float128
  Span<Int128> m_int128_view; //!< Vue les Int128

  ArrayView<Int64> m_sizes_view; //!< Vue pour les tailles (doit être un multiple de ALIGN_SIZE);

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
  Span<Int8> getInt8Buffer() override { return m_int8_view; }
  Span<Float16> getFloat16Buffer() override { return m_float16_view; }
  Span<BFloat16> getBFloat16Buffer() override { return m_bfloat16_view; }
  Span<Float32> getFloat32Buffer() override { return m_float32_view; }
  Span<Float128> getFloat128Buffer() override { return m_float128_view; }
  Span<Int128> getInt128Buffer() override { return m_int128_view; }

  void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                      Int64 nb_int64, Int64 nb_byte) override
  {
    Int64 nb_int8 = 0;
    Int64 nb_float16 = 0;
    Int64 nb_bfloat16 = 0;
    Int64 nb_float32 = 0;
    Int64 nb_float128 = 0;
    Int64 nb_int128 = 0;
    allocateBuffer(nb_real, nb_int16, nb_int32, nb_int64, nb_byte, nb_int8,
                   nb_float16, nb_bfloat16, nb_float32, nb_float128, nb_int128);
  }

  void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                      Int64 nb_int64, Int64 nb_byte, Int64 nb_int8, Int64 nb_float16,
                      Int64 nb_bfloat16, Int64 nb_float32, Int64 nb_float128, Int64 nb_int128) override
  {
    SizeInfo size_info = getPaddingSize(NB_SIZE_ELEM, sizeof(Int64));
    Int64 total = size_info.m_padded_size;

    Int64 real_position = total;
    SizeInfo padded_real_size = getPaddingSize(nb_real, sizeof(Real));
    total += padded_real_size.m_padded_size;

    Int64 int16_position = total;
    SizeInfo padded_int16_size = getPaddingSize(nb_int16, sizeof(Int16));
    total += padded_int16_size.m_padded_size;

    Int64 int32_position = total;
    SizeInfo padded_int32_size = getPaddingSize(nb_int32, sizeof(Int32));
    total += padded_int32_size.m_padded_size;

    Int64 int64_position = total;
    SizeInfo padded_int64_size = getPaddingSize(nb_int64, sizeof(Int64));
    total += padded_int64_size.m_padded_size;

    Int64 byte_position = total;
    SizeInfo padded_byte_size = getPaddingSize(nb_byte, sizeof(Byte));
    total += padded_byte_size.m_padded_size;

    Int64 int8_position = total;
    SizeInfo padded_int8_size = getPaddingSize(nb_int8, sizeof(Int8));
    total += padded_int8_size.m_padded_size;

    Int64 float16_position = total;
    SizeInfo padded_float16_size = getPaddingSize(nb_float16, sizeof(Float16));
    total += padded_float16_size.m_padded_size;

    Int64 bfloat16_position = total;
    SizeInfo padded_bfloat16_size = getPaddingSize(nb_bfloat16, sizeof(BFloat16));
    total += padded_bfloat16_size.m_padded_size;

    Int64 float32_position = total;
    SizeInfo padded_float32_size = getPaddingSize(nb_float32, sizeof(Float32));
    total += padded_float32_size.m_padded_size;

    Int64 float128_position = total;
    SizeInfo padded_float128_size = getPaddingSize(nb_float128, sizeof(Float128));
    total += padded_float128_size.m_padded_size;

    Int64 int128_position = total;
    SizeInfo padded_int128_size = getPaddingSize(nb_int128, sizeof(Int128));
    total += padded_int128_size.m_padded_size;

    _allocBuffer(total);

    _fillPadding(0, size_info);
    _fillPadding(real_position, padded_real_size);
    _fillPadding(int16_position, padded_int16_size);
    _fillPadding(int32_position, padded_int32_size);
    _fillPadding(int64_position, padded_int64_size);
    _fillPadding(byte_position, padded_byte_size);
    _fillPadding(int8_position, padded_int8_size);
    _fillPadding(float16_position, padded_float16_size);
    _fillPadding(bfloat16_position, padded_bfloat16_size);
    _fillPadding(float32_position, padded_float32_size);
    _fillPadding(float128_position, padded_float128_size);
    _fillPadding(int128_position, padded_int128_size);

    m_sizes_view = ArrayView<Int64>(NB_SIZE_ELEM, (Int64*)&m_buffer_view[0]);
    m_sizes_view.fill(0);

    m_sizes_view[IDX_TAG] = SERIALIZE_TAG;
    m_sizes_view[IDX_VERSION] = 1;
    m_sizes_view[IDX_RESERVED1] = 0;

    m_sizes_view[IDX_TOTAL_SIZE] = total;
    m_sizes_view[IDX_NB_FLOAT64] = nb_real;
    m_sizes_view[IDX_NB_INT64] = nb_int64;
    m_sizes_view[IDX_NB_INT32] = nb_int32;
    m_sizes_view[IDX_NB_INT16] = nb_int16;
    m_sizes_view[IDX_NB_BYTE] = nb_byte;
    m_sizes_view[IDX_NB_INT8] = nb_int8;
    m_sizes_view[IDX_NB_FLOAT16] = nb_float16;
    m_sizes_view[IDX_NB_BFLOAT16] = nb_bfloat16;
    m_sizes_view[IDX_NB_FLOAT32] = nb_float32;
    m_sizes_view[IDX_NB_FLOAT128] = nb_float128;
    m_sizes_view[IDX_NB_INT128] = nb_int128;

    m_sizes_view[IDX_POS_FLOAT64] = real_position;
    m_sizes_view[IDX_POS_INT64] = int64_position;
    m_sizes_view[IDX_POS_INT32] = int32_position;
    m_sizes_view[IDX_POS_INT16] = int16_position;
    m_sizes_view[IDX_POS_BYTE] = byte_position;
    m_sizes_view[IDX_POS_INT8] = int8_position;
    m_sizes_view[IDX_POS_FLOAT16] = float16_position;
    m_sizes_view[IDX_POS_BFLOAT16] = bfloat16_position;
    m_sizes_view[IDX_POS_FLOAT32] = float32_position;
    m_sizes_view[IDX_POS_FLOAT128] = float32_position;
    m_sizes_view[IDX_POS_INT128] = int128_position;

    m_real_view = Span<Real>((Real*)&m_buffer_view[real_position], nb_real);
    m_int16_view = Span<Int16>((Int16*)&m_buffer_view[int16_position], nb_int16);
    m_int32_view = Span<Int32>((Int32*)&m_buffer_view[int32_position], nb_int32);
    m_int64_view = Span<Int64>((Int64*)&m_buffer_view[int64_position], nb_int64);
    m_byte_view = Span<Byte>((Byte*)&m_buffer_view[byte_position], nb_byte);
    m_int8_view = Span<Int8>((Int8*)&m_buffer_view[int8_position], nb_int8);
    m_float16_view = Span<Float16>((Float16*)&m_buffer_view[float16_position], nb_float16);
    m_bfloat16_view = Span<BFloat16>((BFloat16*)&m_buffer_view[bfloat16_position], nb_bfloat16);
    m_float32_view = Span<Float32>((Float32*)&m_buffer_view[float32_position], nb_float32);
    m_float128_view = Span<Float128>((Float128*)&m_buffer_view[float128_position], nb_float128);
    m_int128_view = Span<Int128>((Int128*)&m_buffer_view[int128_position], nb_int128);

    _checkAlignment();
  }

  void copy(Impl* rhs) override
  {
    m_real_view.copy(rhs->getRealBuffer());
    m_int64_view.copy(rhs->getInt64Buffer());
    m_int32_view.copy(rhs->getInt32Buffer());
    m_int16_view.copy(rhs->getInt16Buffer());
    m_byte_view.copy(rhs->getByteBuffer());
    m_int8_view.copy(rhs->getInt8Buffer());
    m_float16_view.copy(rhs->getFloat16Buffer());
    m_bfloat16_view.copy(rhs->getBFloat16Buffer());
    m_float32_view.copy(rhs->getFloat32Buffer());
    m_float128_view.copy(rhs->getFloat128Buffer());
    m_int128_view.copy(rhs->getInt128Buffer());

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

  ConstArrayView<Int64> sizesBuffer() const override
  {
    return m_sizes_view;
  }

  void preallocate(Int64 size) override
  {
    _allocBuffer(size);
    m_sizes_view = ArrayView<Int64>(NB_SIZE_ELEM, (Int64*)&m_buffer_view[0]);
  }

  void releaseBuffer() override
  {
    m_buffer.dispose();
  }

  void setFromSizes() override
  {
    Int64 tag_id = m_sizes_view[IDX_TAG];
    if (tag_id != SERIALIZE_TAG)
      ARCCORE_FATAL("Bad tag id '{0}' for serializer (expected={1})."
                    "The data are not from a BasicSerializer. SizeView={2}",
                    tag_id, SERIALIZE_TAG, m_sizes_view);
    Int64 version_id = m_sizes_view[IDX_VERSION];
    if (version_id != 1)
      ARCCORE_FATAL("Bad version '{0}' for serializer. Only version 1 is allowed", version_id);

    Int64 nb_real = m_sizes_view[IDX_NB_FLOAT64];
    Int64 nb_int64 = m_sizes_view[IDX_NB_INT64];
    Int64 nb_int32 = m_sizes_view[IDX_NB_INT32];
    Int64 nb_int16 = m_sizes_view[IDX_NB_INT16];
    Int64 nb_byte = m_sizes_view[IDX_NB_BYTE];
    Int64 nb_int8 = m_sizes_view[IDX_NB_INT8];
    Int64 nb_float16 = m_sizes_view[IDX_NB_FLOAT16];
    Int64 nb_bfloat16 = m_sizes_view[IDX_NB_BFLOAT16];
    Int64 nb_float32 = m_sizes_view[IDX_NB_FLOAT32];
    Int64 nb_float128 = m_sizes_view[IDX_NB_FLOAT128];
    Int64 nb_int128 = m_sizes_view[IDX_NB_INT128];

    Int64 real_position = m_sizes_view[IDX_POS_FLOAT64];
    Int64 int64_position = m_sizes_view[IDX_POS_INT64];
    Int64 int32_position = m_sizes_view[IDX_POS_INT32];
    Int64 int16_position = m_sizes_view[IDX_POS_INT16];
    Int64 byte_position = m_sizes_view[IDX_POS_BYTE];
    Int64 int8_position = m_sizes_view[IDX_POS_INT8];
    Int64 float16_position = m_sizes_view[IDX_POS_FLOAT16];
    Int64 bfloat16_position = m_sizes_view[IDX_POS_BFLOAT16];
    Int64 float32_position = m_sizes_view[IDX_POS_FLOAT32];
    Int64 float128_position = m_sizes_view[IDX_POS_FLOAT128];
    Int64 int128_position = m_sizes_view[IDX_POS_INT128];

    m_real_view = Span<Real>((Real*)&m_buffer_view[real_position], nb_real);
    m_int16_view = Span<Int16>((Int16*)&m_buffer_view[int16_position], nb_int16);
    m_int32_view = Span<Int32>((Int32*)&m_buffer_view[int32_position], nb_int32);
    m_int64_view = Span<Int64>((Int64*)&m_buffer_view[int64_position], nb_int64);
    m_byte_view = Span<Byte>((Byte*)&m_buffer_view[byte_position], nb_byte);
    m_int8_view = Span<Int8>((Int8*)&m_buffer_view[int8_position], nb_int8);
    m_float16_view = Span<Float16>((Float16*)&m_buffer_view[float16_position], nb_float16);
    m_bfloat16_view = Span<BFloat16>((BFloat16*)&m_buffer_view[bfloat16_position], nb_bfloat16);
    m_float32_view = Span<Float32>((Float32*)&m_buffer_view[float32_position], nb_float32);
    m_float128_view = Span<Float128>((Float128*)&m_buffer_view[float128_position], nb_float128);
    m_int128_view = Span<Int128>((Int128*)&m_buffer_view[float128_position], nb_int128);

    _checkAlignment();
  }

  ConstArrayView<Byte> copyAndGetSizesBuffer() override
  {
    // Recopie dans \a m_size_copy_buffer les valeurs de \a m_size_view
    // et retourne un pointeur sur \a m_size_copy_buffer.
    ArrayView<Int64> copy_buf(NB_SIZE_ELEM, m_size_copy_buffer);
    copy_buf.copy(m_sizes_view);
    ConstArrayView<Byte> bytes(sizeof(m_size_copy_buffer), (const Byte*)m_size_copy_buffer);
    return bytes;
  }

  Int64 totalSize() const override
  {
    return m_sizes_view[IDX_TOTAL_SIZE];
  }

  void printSizes(std::ostream& o) const override
  {
    ConstArrayView<Int64> sbuf_sizes = this->sizesBuffer();
    Int64 total_size = totalSize();
    Span<Byte> bytes = m_buffer_view;
    o << " bytes " << bytes.size()
      << " total_size " << total_size
      << " float64 " << sbuf_sizes[IDX_NB_FLOAT64]
      << " int64 " << sbuf_sizes[IDX_NB_INT64]
      << " int32 " << sbuf_sizes[IDX_NB_INT32]
      << " int16 " << sbuf_sizes[IDX_NB_INT16]
      << " byte " << sbuf_sizes[IDX_NB_BYTE]
      << " int8 " << sbuf_sizes[IDX_NB_INT8]
      << " float16 " << sbuf_sizes[IDX_NB_FLOAT16]
      << " bfloat16 " << sbuf_sizes[IDX_NB_BFLOAT16]
      << " float32 " << sbuf_sizes[IDX_NB_FLOAT32]
      << " float128 " << sbuf_sizes[IDX_NB_FLOAT128]
      << " int128 " << sbuf_sizes[IDX_NB_INT128]
      << " ptr=" << (void*)bytes.data();
  }

 protected:

  SizeInfo getPaddingSize(Int64 nb_elem, Int64 elem_size)
  {
    if (nb_elem < 0)
      ARCCORE_FATAL("Bad number of element '{0}' (should be >=0)", nb_elem);
    if (elem_size <= 0)
      ARCCORE_FATAL("Bad elem_size '{0}'", elem_size);
    Int64 s = nb_elem * elem_size;
    Int64 pad = s % ALIGN_SIZE;
    if (pad == 0)
      pad = ALIGN_SIZE;
    Int64 new_size = s + (ALIGN_SIZE - pad);
    if ((new_size % ALIGN_SIZE) != 0)
      ARCCORE_FATAL("Bad padding {0}", new_size);
    //std::cout << " nb_elem=" << nb_elem << " elem_size=" << elem_size << " s=" << s << " new_size=" << new_size << '\n';
    return { s, new_size };
  }

  /*!
   * \brief Remplit avec une valeur fixe les zones correspondantes au padding.
   * Cela permet d'éviter d'avoir des valeurs non initialisées.
   *
   * Il faut avoir appeler _allocBuffer() avant
   */
  void _fillPadding(Int64 position, SizeInfo size_info)
  {
    Int64 begin = position + size_info.m_original_size;
    Int64 end = position + size_info.m_padded_size;
    _fillPadding(m_buffer_view.subspan(begin, end - begin));
  }

  void _fillPadding(Span<Byte> buf)
  {
    // Utilise une valeur non nulle pour repérer plus facilement
    // les zones de padding si besoin.
    constexpr Byte v = (Byte)(250);
    for (Int64 i = 0, s = buf.size(); i < s; ++i)
      buf[i] = v;
  }

  void _checkAlignment()
  {
    _checkAddr(m_real_view.data());
    _checkAddr(m_int16_view.data());
    _checkAddr(m_int32_view.data());
    _checkAddr(m_int64_view.data());
    _checkAddr(m_byte_view.data());
    _checkAddr(m_int8_view.data());
    _checkAddr(m_float16_view.data());
    _checkAddr(m_bfloat16_view.data());
    _checkAddr(m_float32_view.data());
    _checkAddr(m_float128_view.data());
    _checkAddr(m_int128_view.data());
  }

  void _checkAddr(void* ptr)
  {
    Int64 addr = (Int64)ptr;
    if ((addr % ALIGN_SIZE) != 0) {
      _printAlignment();
      ARCCORE_FATAL("Bad alignment addr={0} - {1}", addr, (addr % ALIGN_SIZE));
    }
  }

  void _printAlignment()
  {
    for (Integer i = 0, n = m_sizes_view.size(); i < n; ++i)
      std::cout << " Size i=" << i << " v=" << m_sizes_view[i] << " pad=" << (m_sizes_view[i] % ALIGN_SIZE) << '\n';
    _printAddr(m_buffer_view.data(), "Buffer");
    _printAddr(m_real_view.data(), "Real");
    _printAddr(m_int16_view.data(), "Int16");
    _printAddr(m_int32_view.data(), "Int32");
    _printAddr(m_int64_view.data(), "Int64");
    _printAddr(m_byte_view.data(), "Byte");
    _printAddr(m_int8_view.data(), "Int8");
    _printAddr(m_float16_view.data(), "Float16");
    _printAddr(m_bfloat16_view.data(), "BFloat16");
    _printAddr(m_float32_view.data(), "Float32");
    _printAddr(m_float128_view.data(), "Float128");
    _printAddr(m_int128_view.data(), "Int128");
  }

  void _printAddr(void* ptr, const String& name)
  {
    Int64 addr = (Int64)ptr;
    std::cout << "Align type=" << name << " addr=" << addr << " offset=" << (addr % ALIGN_SIZE) << '\n';
  }

  void _allocBuffer(Int64 size)
  {
    if (size < 1024)
      size = 1024;
    m_buffer.resize(size + ALIGN_SIZE * 4);
    Int64 addr = (Int64)(&m_buffer[0]);
    Int64 padding = addr % ALIGN_SIZE;
    Int64 position = 0;
    if (padding != 0) {
      position = ALIGN_SIZE - padding;
    }
    // La taille doit être un multiple de ALIGN_SIZE;
    Int64 new_size = (size + ALIGN_SIZE) - (size % ALIGN_SIZE);
    m_buffer_view = m_buffer.span().subspan(position, new_size);

    // Initialise les valeurs de la zone tampon
    auto padding_view = m_buffer.span().subspan(position + size, new_size - size);
    _fillPadding(padding_view);
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
reserve(eDataType dt, Int64 n, Int64 nb_put)
{
  ARCCORE_ASSERT((m_mode == ModeReserve), ("Bad mode"));
  switch (dt) {
  case DT_Real:
    m_real.m_reserved_size += n;
    break;
  case DT_Int64:
    m_int64.m_reserved_size += n;
    break;
  case DT_Int32:
    m_int32.m_reserved_size += n;
    break;
  case DT_Int16:
    m_int16.m_reserved_size += n;
    break;
  case DT_Byte:
    m_byte.m_reserved_size += n;
    break;
  case DT_Int8:
    m_int8.m_reserved_size += n;
    break;
  case DT_Float16:
    m_float16.m_reserved_size += n;
    break;
  case DT_BFloat16:
    m_bfloat16.m_reserved_size += n;
    break;
  case DT_Float32:
    m_float32.m_reserved_size += n;
    break;
  case DT_Float128:
    m_float128.m_reserved_size += n;
    break;
  case DT_Int128:
    m_int128.m_reserved_size += n;
    break;
  default:
    ARCCORE_THROW(ArgumentException, "bad datatype v={0}", (int)dt);
  }
  if (m_is_serialize_typeinfo)
    // Pour le type de la donnée.
    m_byte.m_reserved_size += nb_put;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
reserve(eBasicDataType bdt, Int64 n, Int64 nb_put)
{
  ARCCORE_ASSERT((m_mode == ModeReserve), ("Bad mode"));
  switch (bdt) {
  case eBasicDataType::Float64:
    m_real.m_reserved_size += n;
    break;
  case eBasicDataType::Int64:
    m_int64.m_reserved_size += n;
    break;
  case eBasicDataType::Int32:
    m_int32.m_reserved_size += n;
    break;
  case eBasicDataType::Int16:
    m_int16.m_reserved_size += n;
    break;
  case eBasicDataType::Byte:
    m_byte.m_reserved_size += n;
    break;
  case eBasicDataType::Int8:
    m_int8.m_reserved_size += n;
    break;
  case eBasicDataType::Float16:
    m_float16.m_reserved_size += n;
    break;
  case eBasicDataType::BFloat16:
    m_bfloat16.m_reserved_size += n;
    break;
  case eBasicDataType::Float32:
    m_float32.m_reserved_size += n;
    break;
  case eBasicDataType::Float128:
    m_float128.m_reserved_size += n;
    break;
  case eBasicDataType::Int128:
    m_int128.m_reserved_size += n;
    break;
  default:
    ARCCORE_THROW(ArgumentException, "Bad basic datatype v={0}", (int)bdt);
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
  if (m_is_serialize_typeinfo) {
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
  m_byte.get(Span<Byte>(&b, 1));
  eBasicDataType t = static_cast<eBasicDataType>(b);
  if (t != expected_type)
    ARCCORE_FATAL("Bad serialized type t='{0}' int={1}' expected='{2}'", t, (int)t, expected_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
allocateBuffer()
{
  m_p->allocateBuffer(m_real.m_reserved_size, m_int16.m_reserved_size, m_int32.m_reserved_size,
                      m_int64.m_reserved_size, m_byte.m_reserved_size, m_int8.m_reserved_size,
                      m_float16.m_reserved_size, m_bfloat16.m_reserved_size, m_float32.m_reserved_size,
                      m_float128.m_reserved_size, m_int128.m_reserved_size);
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

  m_int8.m_buffer = m_p->getInt8Buffer();
  m_int8.m_current_position = 0;

  m_float16.m_buffer = m_p->getFloat16Buffer();
  m_float16.m_current_position = 0;

  m_bfloat16.m_buffer = m_p->getBFloat16Buffer();
  m_bfloat16.m_current_position = 0;

  m_float32.m_buffer = m_p->getFloat32Buffer();
  m_float32.m_current_position = 0;

  m_float128.m_buffer = m_p->getFloat128Buffer();
  m_float128.m_current_position = 0;

  m_int128.m_buffer = m_p->getInt128Buffer();
  m_int128.m_current_position = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
               Int64 nb_int64, Int64 nb_byte)
{
  Int64 nb_int8 = 0;
  Int64 nb_float16 = 0;
  Int64 nb_bfloat16 = 0;
  Int64 nb_float32 = 0;
  Int64 nb_float128 = 0;
  Int64 nb_int128 = 0;
  allocateBuffer(nb_real, nb_int16, nb_int32, nb_int64, nb_byte, nb_int8,
                 nb_float16, nb_bfloat16, nb_float32, nb_float128, nb_int128);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
               Int64 nb_int64, Int64 nb_byte, Int64 nb_int8, Int64 nb_float16,
               Int64 nb_bfloat16, Int64 nb_float32, Int64 nb_float128, Int64 nb_int128)
{
  m_real.m_reserved_size = nb_real;
  m_int64.m_reserved_size = nb_int64;
  m_int32.m_reserved_size = nb_int32;
  m_int16.m_reserved_size = nb_int16;
  m_byte.m_reserved_size = nb_byte;
  m_int8.m_reserved_size = nb_int8;
  m_float16.m_reserved_size = nb_float16;
  m_bfloat16.m_reserved_size = nb_bfloat16;
  m_float32.m_reserved_size = nb_float32;
  m_float128.m_reserved_size = nb_float128;
  m_int128.m_reserved_size = nb_int128;
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
  Span<Int8> int8_b = rhs_p->getInt8Buffer();
  Span<Float16> float16_b = rhs_p->getFloat16Buffer();
  Span<BFloat16> bfloat16_b = rhs_p->getBFloat16Buffer();
  Span<Float32> float32_b = rhs_p->getFloat32Buffer();
  Span<Float128> float128_b = rhs_p->getFloat128Buffer();
  Span<Int128> int128_b = rhs_p->getInt128Buffer();
  allocateBuffer(real_b.size(), int16_b.size(), int32_b.size(), int64_b.size(), byte_b.size(),
                 int8_b.size(), float16_b.size(), bfloat16_b.size(),
                 float32_b.size(), float128_b.size(), int128_b.size());
  m_p->copy(rhs_p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::Impl2::
setMode(eMode new_mode)
{
  if (new_mode == BasicSerializer::ModeGet && new_mode != m_mode) {
    m_real.m_current_position = 0;
    m_int64.m_current_position = 0;
    m_int32.m_current_position = 0;
    m_int16.m_current_position = 0;
    m_byte.m_current_position = 0;
    m_int8.m_current_position = 0;
    m_float16.m_current_position = 0;
    m_bfloat16.m_current_position = 0;
    m_float32.m_current_position = 0;
    m_float128.m_current_position = 0;
    m_int128.m_current_position = 0;
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
Span<Real> BasicSerializer::realBuffer()
{
  return m_p2->m_real.m_buffer;
}
Span<Int64> BasicSerializer::int64Buffer()
{
  return m_p2->m_int64.m_buffer;
}
Span<Int32> BasicSerializer::int32Buffer()
{
  return m_p2->m_int32.m_buffer;
}
Span<Int16> BasicSerializer::int16Buffer()
{
  return m_p2->m_int16.m_buffer;
}
Span<Byte> BasicSerializer::byteBuffer()
{
  return m_p2->m_byte.m_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserveSpan(eDataType dt, Int64 n)
{
  m_p2->reserve(dt, n, 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserveSpan(eBasicDataType dt, Int64 n)
{
  m_p2->reserve(dt, n, 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserve(eDataType dt, Int64 n)
{
  m_p2->reserve(dt, n, n);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserve(eBasicDataType dt, Int64 n)
{
  m_p2->reserve(dt, n, n);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserve(const String& str)
{
  reserveInt64(1);
  reserveSpan(str.bytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
reserveArray(Span<const Real> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int16> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int32> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int64> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Byte> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int8> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Float16> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const BFloat16> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Float32> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Float128> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

void BasicSerializer::
reserveArray(Span<const Int128> values)
{
  reserveInt64(1);
  reserveSpan(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Real> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Float64);
  m_p2->m_real.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Int64> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Int64);
  m_p2->m_int64.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Int32> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Int32);
  m_p2->m_int32.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Int16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Int16);
  m_p2->m_int16.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(Span<const Byte> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Byte);
  m_p2->m_byte.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
putSpan(Span<const Int8> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Int8);
  m_p2->m_int8.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
putSpan(Span<const Float16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Float16);
  m_p2->m_float16.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
putSpan(Span<const BFloat16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::BFloat16);
  m_p2->m_bfloat16.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
putSpan(Span<const Float32> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Float32);
  m_p2->m_float32.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
putSpan(Span<const Float128> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Float128);
  m_p2->m_float128.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
putSpan(Span<const Int128> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
  m_p2->putType(eBasicDataType::Int128);
  m_p2->m_int128.put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
put(const String& str)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModePut), ("Bad mode"));
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

void BasicSerializer::
putArray(Span<const Int8> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Float16> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const BFloat16> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Float32> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Float128> values)
{
  putInt64(values.size());
  putSpan(values);
}

void BasicSerializer::
putArray(Span<const Int128> values)
{
  putInt64(values.size());
  putSpan(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Real> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Float64);
  m_p2->m_real.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int64> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int64);
  m_p2->m_int64.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int32> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int32);
  m_p2->m_int32.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int16);
  m_p2->m_int16.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Byte> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Byte);
  m_p2->m_byte.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int8> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int8);
  m_p2->m_int8.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Float16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Float16);
  m_p2->m_float16.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<BFloat16> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::BFloat16);
  m_p2->m_bfloat16.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Float32> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Float32);
  m_p2->m_float32.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Float128> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Float128);
  m_p2->m_float128.get(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializer::
getSpan(Span<Int128> values)
{
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
  m_p2->getAndCheckType(eBasicDataType::Int128);
  m_p2->m_int128.get(values);
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

void BasicSerializer::
getArray(Array<Int8>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Float16>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<BFloat16>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Float32>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Float128>& values)
{
  values.resize(getInt64());
  getSpan(values);
}

void BasicSerializer::
getArray(Array<Int128>& values)
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
  ARCCORE_ASSERT((m_p2->m_mode == ModeGet), ("Bad mode"));
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
allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
               Int64 nb_int64, Int64 nb_byte)
{
  m_p2->allocateBuffer(nb_real, nb_int16, nb_int32, nb_int64, nb_byte, 0, 0, 0, 0, 0, 0);
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

ConstArrayView<Byte> BasicSerializer::
copyAndGetSizesBuffer()
{
  return _p()->copyAndGetSizesBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int64> BasicSerializer::
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
releaseBuffer()
{
  _p()->releaseBuffer();
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
