// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicSerializer.h                                           (C) 2000-2025 */
/*                                                                           */
/* Implémentation simple de 'ISerializer'.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_BASICSERIALIZER_H
#define ARCCORE_SERIALIZE_BASICSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/ISerializer.h"

#include "arccore/base/Float16.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/Float128.h"
#include "arccore/base/Int128.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class BasicSerializeGatherMessage;

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
  friend Arcane::BasicSerializeGatherMessage;
  typedef BasicSerializer ThatClass;
  using ISerializer::reserveSpan;
  using ISerializer::putSpan;

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
  void reserveSpan(eBasicDataType dt, Int64 n) override;
  void reserve(eBasicDataType dt, Int64 n) override;
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
  void reserveArray(Span<const Int8> values) override;
  void reserveArray(Span<const Float16> values) override;
  void reserveArray(Span<const BFloat16> values) override;
  void reserveArray(Span<const Float32> values) override;
  void reserveArray(Span<const Float128> values) override;
  void reserveArray(Span<const Int128> values) override;

  void put(Span<const Real> values) override;
  void put(Span<const Int16> values) override;
  void put(Span<const Int32> values) override;
  void put(Span<const Int64> values) override;
  void put(Span<const Byte> values) override;
  void putSpan(Span<const Int8> values) override;
  void putSpan(Span<const Float16> values) override;
  void putSpan(Span<const BFloat16> values) override;
  void putSpan(Span<const Float32> values) override;
  void putSpan(Span<const Float128> values) override;
  void putSpan(Span<const Int128> values) override;
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
  void put(Int8 value) override
  {
    putInt8(value);
  }
  void put(Float16 value) override
  {
    putFloat16(value);
  }
  void put(BFloat16 value) override
  {
    putBFloat16(value);
  }
  void put(Float32 value) override
  {
    putFloat32(value);
  }
  void put(Float128 value) override
  {
    putFloat128(value);
  }
  void put(Int128 value) override
  {
    putInt128(value);
  }

  void putReal(Real value) override
  {
    put(ConstArrayView<Real>(1, &value));
  }
  void putInt64(Int64 value) override
  {
    put(ConstArrayView<Int64>(1, &value));
  }
  void putInt32(Int32 value) override
  {
    put(ConstArrayView<Int32>(1, &value));
  }
  void putInt16(Int16 value) override
  {
    put(ConstArrayView<Int16>(1, &value));
  }
  void putInteger(Integer value) override
  {
#ifdef ARCANE_64BIT
    put(ConstArrayView<Int64>(1, &value));
#else
    put(ConstArrayView<Int32>(1, &value));
#endif
  }
  void putByte(Byte value) override
  {
    put(ConstArrayView<Byte>(1, &value));
  }
  void putInt8(Int8 value) override
  {
    putSpan(ConstArrayView<Int8>(1, &value));
  }
  void putFloat16(Float16 value) override
  {
    putSpan(ConstArrayView<Float16>(1, &value));
  }
  void putBFloat16(BFloat16 value) override
  {
    putSpan(ConstArrayView<BFloat16>(1, &value));
  }
  void putFloat32(Float32 value) override
  {
    putSpan(ConstArrayView<Float32>(1, &value));
  }
  void putFloat128(Float128 value) override
  {
    putSpan(ConstArrayView<Float128>(1, &value));
  }
  void putInt128(Int128 value) override
  {
    putSpan(ConstArrayView<Int128>(1, &value));
  }

  void putArray(Span<const Real> values) override;
  void putArray(Span<const Int16> values) override;
  void putArray(Span<const Int32> values) override;
  void putArray(Span<const Int64> values) override;
  void putArray(Span<const Byte> values) override;
  void putArray(Span<const Int8> values) override;
  void putArray(Span<const Float16> values) override;
  void putArray(Span<const BFloat16> values) override;
  void putArray(Span<const Float32> values) override;
  void putArray(Span<const Float128> values) override;
  void putArray(Span<const Int128> values) override;

  void get(ArrayView<Real> values) override { ThatClass::getSpan(values); }
  void get(ArrayView<Int64> values) override { ThatClass::getSpan(values); }
  void get(ArrayView<Int32> values) override { ThatClass::getSpan(values); }
  void get(ArrayView<Int16> values) override { ThatClass::getSpan(values); }
  void get(ArrayView<Byte> values) override { ThatClass::getSpan(values); }

  void getSpan(Span<Real> values) override;
  void getSpan(Span<Int16> values) override;
  void getSpan(Span<Int32> values) override;
  void getSpan(Span<Int64> values) override;
  void getSpan(Span<Byte> values) override;
  void getSpan(Span<Int8> values) override;
  void getSpan(Span<Float16> values) override;
  void getSpan(Span<BFloat16> values) override;
  void getSpan(Span<Float32> values) override;
  void getSpan(Span<Float128> values) override;
  void getSpan(Span<Int128> values) override;

  void getArray(Array<Real>& values) override;
  void getArray(Array<Int16>& values) override;
  void getArray(Array<Int32>& values) override;
  void getArray(Array<Int64>& values) override;
  void getArray(Array<Byte>& values) override;
  void getArray(Array<Int8>& values) override;
  void getArray(Array<Float16>& values) override;
  void getArray(Array<BFloat16>& values) override;
  void getArray(Array<Float32>& values) override;
  void getArray(Array<Float128>& values) override;
  void getArray(Array<Int128>& values) override;

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
  Int8 getInt8() override
  {
    Int8 r = 0;
    getSpan(ArrayView<Int8>(1, &r));
    return r;
  }
  Float16 getFloat16() override
  {
    Float16 r = {};
    getSpan(ArrayView<Float16>(1, &r));
    return r;
  }
  BFloat16 getBFloat16() override
  {
    BFloat16 r = {};
    getSpan(ArrayView<BFloat16>(1, &r));
    return r;
  }
  Float32 getFloat32() override
  {
    Float32 r = {};
    getSpan(ArrayView<Float32>(1, &r));
    return r;
  }
  Float128 getFloat128() override
  {
    Float128 r = {};
    getSpan(ArrayView<Float128>(1, &r));
    return r;
  }
  Int128 getInt128() override
  {
    Int128 r = {};
    getSpan(ArrayView<Int128>(1, &r));
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

  ConstArrayView<Byte> copyAndGetSizesBuffer();
  Span<Byte> globalBuffer();
  Span<const Byte> globalBuffer() const;
  ARCCORE_DEPRECATED_2020("Do not use. get total size with totalSize()")
  ConstArrayView<Int64> sizesBuffer();
  Int64 totalSize() const;
  void preallocate(Int64 size);
  void releaseBuffer();
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
