﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicSerializer.h                                           (C) 2000-2020 */
/*                                                                           */
/* Implémentation simple de 'ISerializer'.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_BASICSERIALIZER_H
#define ARCCORE_SERIALIZE_BASICSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/ISerializer.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tampon pour sérialiser un type de donnée \a DataType.
 */
template <class DataType>
class BasicSerializerDataT
{
 public:
  BasicSerializerDataT()
  : m_reserved_size(0)
  , m_current_position(0)
  {}

 public:
  void put(Span<const DataType> values)
  {
    Int64 n = values.size();
    Int64 cp = m_current_position;
    Int64 max_size = 1 + m_buffer.size();
    arccoreCheckAt(n + cp, max_size);

    DataType* ptr = m_buffer.data() + cp;
    const DataType* vptr = values.data();
    for (Int64 i = 0; i < n; ++i)
      ptr[i] = vptr[i];
    m_current_position += n;
  }

  void get(Span<DataType> values)
  {
    Int64 n = values.size();
    Int64 cp = m_current_position;
    Int64 max_size = 1 + m_buffer.size();
    arccoreCheckAt(n + cp, max_size);

    const DataType* ptr = m_buffer.data() + cp;
    DataType* vptr = values.data();
    for (Int64 i = 0; i < n; ++i)
      vptr[i] = ptr[i];
    m_current_position += n;
  }

 public:
  Int64 m_reserved_size;
  Int64 m_current_position;
  Span<DataType> m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation basique de 'ISerializer'
 */
class ARCCORE_SERIALIZE_EXPORT BasicSerializer
: public ISerializer
{
  typedef BasicSerializer ThatClass;
  using ISerializer::reserveSpan;
 public:
  // Classe temporaire pour afficher les tailles des buffers.
  class SizesPrinter
  {
   public:
    explicit SizesPrinter(const BasicSerializer& sbuf) : m_sbuf(sbuf){}
    const BasicSerializer& buf() const { return m_sbuf; }
   private:
    const BasicSerializer& m_sbuf;
  };
 public:
  class Impl;
  class Impl2;

 public:
  BasicSerializer();
  BasicSerializer(const BasicSerializer& sb);
  ~BasicSerializer() override;

 private:
  void operator=(const BasicSerializer& sb);

 public:

  using ISerializer::reserve;

 public:
  void reserveSpan(eDataType dt, Int64 n) override;
  void reserve(eDataType dt, Int64 n) override;
  void reserveInteger(Int64 n) override
  {
#ifdef ARCANE_64BIT
    reserve(DT_Int64, n);
#else
    reserve(DT_Int32, n);
#endif
  }
  void reserve(const String& str) override;

  void reserveArray(Span<const Real> values) override;
  void reserveArray(Span<const Int16> values) override;
  void reserveArray(Span<const Int32> values) override;
  void reserveArray(Span<const Int64> values) override;
  void reserveArray(Span<const Byte> values) override;

  void put(Span<const Real> values) override;
  void put(Span<const Int16> values) override;
  void put(Span<const Int32> values) override;
  void put(Span<const Int64> values) override;
  void put(Span<const Byte> values) override;
  void put(const String& value) override;

  void put(Real value) override
  {
    putReal(value);
  }
  void put(Int64 value) override
  {
    putInt64(value);
  }
  void put(Int32 value) override
  {
    putInt32(value);
  }
  void put(Int16 value) override
  {
    putInt16(value);
  }
  void put(Byte value) override
  {
    putByte(value);
  }

  void putReal(Real value) override
  {
    put(RealConstArrayView(1, &value));
  }
  void putInt64(Int64 value) override
  {
    put(Int64ConstArrayView(1, &value));
  }
  void putInt32(Int32 value) override
  {
    put(Int32ConstArrayView(1, &value));
  }
  void putInt16(Int16 value) override
  {
    put(Int16ConstArrayView(1, &value));
  }
  void putInteger(Integer value) override
  {
#ifdef ARCANE_64BIT
    put(Int64ConstArrayView(1, &value));
#else
    put(Int32ConstArrayView(1, &value));
#endif
  }
  void putByte(Byte value) override
  {
    put(ByteConstArrayView(1, &value));
  }

  void putArray(Span<const Real> values) override;
  void putArray(Span<const Int16> values) override;
  void putArray(Span<const Int32> values) override;
  void putArray(Span<const Int64> values) override;
  void putArray(Span<const Byte> values) override;

  void get(RealArrayView values) override { ThatClass::getSpan(values); }
  void get(Int64ArrayView values) override { ThatClass::getSpan(values); }
  void get(Int32ArrayView values) override { ThatClass::getSpan(values); }
  void get(Int16ArrayView values) override { ThatClass::getSpan(values); }
  void get(ByteArrayView values) override { ThatClass::getSpan(values); }

  void getSpan(Span<Real> values) override;
  void getSpan(Span<Int16> values) override;
  void getSpan(Span<Int32> values) override;
  void getSpan(Span<Int64> values) override;
  void getSpan(Span<Byte> values) override;

  void getArray(Array<Real>& values) override;
  void getArray(Array<Int16>& values) override;
  void getArray(Array<Int32>& values) override;
  void getArray(Array<Int64>& values) override;
  void getArray(Array<Byte>& values) override;

  void get(String& values) override;

  Real getReal() override
  {
    Real r = 0.;
    get(ArrayView<Real>(1, &r));
    return r;
  }
  Int64 getInt64() override
  {
    Int64 r = 0;
    get(ArrayView<Int64>(1, &r));
    return r;
  }
  Int32 getInt32() override
  {
    Int32 r = 0;
    get(ArrayView<Int32>(1, &r));
    return r;
  }
  Int16 getInt16() override
  {
    Int16 r = 0;
    get(ArrayView<Int16>(1, &r));
    return r;
  }
  Integer getInteger() override
  {
#ifdef ARCANE_64BIT
    return getInt64();
#else
    return getInt32();
#endif
  }
  Byte getByte() override
  {
    Byte r = 0;
    get(ArrayView<Byte>(1, &r));
    return r;
  }

  void allocateBuffer() override;

  eMode mode() const override;
  void setMode(eMode new_mode) override;
  eReadMode readMode() const override;
  void setReadMode(eReadMode new_read_mode) override;

 public:
  /*!
   * \brief Indique si on sérialise le type de donnée pour
   * garantir la cohérence.
   *
   * Si actif, cela nécessite que les appels à reserve() et
   * reserveArray() soient cohérents avec les put(). Comme ce
   * n'est pas le cas historiquement, cette option n'est pas active
   * par défaut.
   *
   * Il n'est utile de positionner cette option qu'en écriture. En lecture,
   * l'information est contenue dans le sérialiseur.
   */  
  void setSerializeTypeInfo(bool v);
  bool isSerializeTypeInfo() const;

 private:

  // Méthode obsolète dans l'interface. A supprimer dès que possible
  void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                      Int64 nb_int64, Int64 nb_byte) override;

 public:

  ARCCORE_DEPRECATED_2020("internal method")
  Span<Real> realBuffer();
  ARCCORE_DEPRECATED_2020("internal method")
  Span<Int64> int64Buffer();
  ARCCORE_DEPRECATED_2020("internal method")
  Span<Int32> int32Buffer();
  ARCCORE_DEPRECATED_2020("internal method")
  Span<Int16> int16Buffer();
  ARCCORE_DEPRECATED_2020("internal method")
  Span<Byte> byteBuffer();

 public:

  ByteConstArrayView copyAndGetSizesBuffer();
  Span<Byte> globalBuffer();
  Span<const Byte> globalBuffer() const;
  ARCCORE_DEPRECATED_2020("Do not use. get total size with totalSize()")
  Int64ConstArrayView sizesBuffer();
  Int64 totalSize() const;
  void preallocate(Int64 size);
  void setFromSizes();
  void printSizes(std::ostream& o) const;

  friend inline std::ostream&
  operator<<(std::ostream& o,const BasicSerializer::SizesPrinter& x)
  {
    x.buf().printSizes(o);
    return o;
  }

 public:

  /*!
   * \brief Initialise le sérialiseur en lecture à partir des données \a buf.
   *
   * Le tableau \a buf doit avoir été obtenu via l'appel à globalBuffer()
   * d'un sérialiseur en écriture.
   */
  void initFromBuffer(Span<const Byte> buf);
  void copy(const ISerializer* from) override;
  void copy(const BasicSerializer& rhs);
  /*!
   * \brief Taille du padding et de l'alignement.
   *
   * Il est garanti que chaque tableau (buffer) géré par ce sérialiseur
   * a une taille en octet multiple de paddingSize() et un
   * alignement sur paddingSize().
   */
  static ARCCORE_CONSTEXPR Integer paddingSize() { return 128; }

  // TEMPORAIRE tant qu'on utilise le AllGather de Arcane. Ensuite à mettre privé
 protected:

  Impl2* m_p2;

  Impl* _p() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
