// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayVariant.h                                              (C) 2000-2025 */
/*                                                                           */
/* Type de base polymorphe pour les tableaux mono-dim (dimension 1).         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATATYPE_ARRAYVARIANT_H
#define ARCANE_CORE_DATATYPE_ARRAYVARIANT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Iostream.h"

#include "arcane/core/datatype/VariantBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Real2;
class Real3;
class Real2x2;
class Real3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Type de base polymorphe pour les tableaux (dimension 1).
 *
 */
class ARCANE_DATATYPE_EXPORT ArrayVariant
: public VariantBase
{
 public:
  ArrayVariant(eType type,Integer size);
  ArrayVariant(ArrayView<Real> data);
  ArrayVariant(ArrayView<Real2> data);
  ArrayVariant(ArrayView<Real3> data);
  ArrayVariant(ArrayView<Real2x2> data);
  ArrayVariant(ArrayView<Real3x3> data);
  ArrayVariant(ArrayView<Int32> data);
  ArrayVariant(ArrayView<Int64> data);
  ArrayVariant(ArrayView<bool> data);
  ArrayVariant(ArrayView<String> data);
  ~ArrayVariant();

 public:

  Integer size() const;

  void value(ArrayView<Real>& v) const { v = m_real_value; }
  void value(ArrayView<Real2>& v) const { v = m_real2_value; }
  void value(ArrayView<Real3>& v) const { v = m_real3_value; }
  void value(ArrayView<Real2x2>& v) const { v = m_real2x2_value; }
  void value(ArrayView<Real3x3>& v) const { v = m_real3x3_value; }
  void value(ArrayView<Int32>& v) const { v = m_int32_value; }
  void value(ArrayView<Int64>& v) const { v = m_int64_value; }
  void value(ArrayView<bool>& v) const { v = m_bool_value; }
  void value(ArrayView<String>& v) const { v = m_string_value; }

  ArrayView<Real> asReal() { return m_real_value; }
  ArrayView<Real2> asReal2() { return m_real2_value; }
  ArrayView<Real3> asReal3() { return m_real3_value; }
  ArrayView<Real2x2> asReal2x2() { return m_real2x2_value; }
  ArrayView<Real3x3> asReal3x3() { return m_real3x3_value; }
  ArrayView<Integer> asInteger();
  ArrayView<Int32> asInt32() { return m_int32_value; }
  ArrayView<bool> asBool() { return m_bool_value; }
  ArrayView<String> asString() { return m_string_value; }

  ConstArrayView<Real> asReal() const { return m_real_value; }
  ConstArrayView<Real2> asReal2() const { return m_real2_value; }
  ConstArrayView<Real3> asReal3() const { return m_real3_value; }
  ConstArrayView<Real2x2> asReal2x2() const { return m_real2x2_value; }
  ConstArrayView<Real3x3> asReal3x3() const { return m_real3x3_value; }
  ConstArrayView<Integer> asInteger() const;
  ConstArrayView<Int32> asInt32() const { return m_int32_value; }
  ConstArrayView<Int64> asInt64() const { return m_int64_value; }
  ConstArrayView<bool> asBool() const { return m_bool_value; }
  ConstArrayView<String> asString() const { return m_string_value; }

 public:


 private:
  ArrayView<Real> m_real_value; //!< Valeur de type tableau de reels
  ArrayView<Real2> m_real2_value; //!< Valeur de type tableau de Real2
  ArrayView<Real3> m_real3_value; //!< Valeur de type tableau de Real3
  ArrayView<Real2x2> m_real2x2_value; //!< Valeur de type tableau de Real2x2
  ArrayView<Real3x3> m_real3x3_value; //!< Valeur de type tableau de Real3x3
  ArrayView<Int32> m_int32_value; //!< Valeur de type tableau d'entiers 32 bits
  ArrayView<Int64> m_int64_value; //!< Valeur de type tableau d'entiers 64 bits
  ArrayView<bool> m_bool_value; //!< Valeur de type tableau de booleens
  ArrayView<String> m_string_value; //!< Valeur de type tableau de chaines
  void* m_allocated_array; //!< Non nul si tableau alloué par le variant

  void _destroy();
};

ARCANE_DATATYPE_EXPORT std::ostream&
operator<<(std::ostream& s, const ArrayVariant& x);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
