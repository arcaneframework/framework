// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScalarVariant.h                                             (C) 2000-2025 */
/*                                                                           */
/* Type de base polymorphe pour les scalaires (dimension 0).                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATATYPE_SCALARVARIANT_H
#define ARCANE_CORE_DATATYPE_SCALARVARIANT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/core/datatype/VariantBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Type de base polymorphe pour les scalaires (dimension 0).
 */
class ARCANE_DATATYPE_EXPORT ScalarVariant 
: public VariantBase
{
 private:

 public:
  ScalarVariant();
  ScalarVariant(const ScalarVariant& v);
  ScalarVariant(Real v);
  ScalarVariant(Real2 v);
  ScalarVariant(Real3 v);
  ScalarVariant(Real2x2 v);
  ScalarVariant(Real3x3 v);
  ScalarVariant(Int32 v);
  ScalarVariant(Int64 v);
  ScalarVariant(bool v);
  ScalarVariant(const String& v);
  ~ScalarVariant() {}

  ScalarVariant& operator= (const ScalarVariant& v);
  
  bool isInteger() const { return m_type==TInt32 || m_type==TInt64; }
  
  void setValue(Real v) { m_real_value = v; m_type = TReal; }
  void setValue(Real2 v) { m_real2_value = v; m_type = TReal2; }
  void setValue(Real3 v) { m_real3_value = v; m_type = TReal3; }
  void setValue(Real2x2 v) { m_real2x2_value = v; m_type = TReal2x2; }
  void setValue(Real3x3 v) { m_real3x3_value = v; m_type = TReal3x3; }
  void setValue(Int32 v) { m_int32_value = v; m_type = TInt32; }
  void setValue(Int64 v) { m_int64_value = v; m_type = TInt64; }
  void setValue(bool v) { m_bool_value = v; m_type = TBool; }
  void setValue(const String& v) { m_string_value = v; m_type = TString; }

  void value(Real& v) const { v = m_real_value; }
  void value(Real2& v) const { v = m_real2_value; }
  void value(Real3& v) const { v = m_real3_value; }
  void value(Real2x2& v) const { v = m_real2x2_value; }
  void value(Real3x3& v) const { v = m_real3x3_value; }
  void value(Int32& v) const { v = m_int32_value; }
  void value(Int64& v) const { v = m_int64_value; }
  void value(bool& v) const { v = m_bool_value; }
  void value(String& v) const { v = m_string_value; }

  Real asReal() const { return m_real_value; }
  Real2 asReal2() const { return m_real2_value; }
  Real3 asReal3() const { return m_real3_value; }
  Real2x2 asReal2x2() const { return m_real2x2_value; }
  Real3x3 asReal3x3() const { return m_real3x3_value; }
  Integer asInteger() const;
  Int32 asInt32() const { return m_int32_value; }
  Int64 asInt64() const { return m_int64_value; }
  bool asBool() const { return m_bool_value; }
  const String& asString() const { return m_string_value; }
  
 private:

  Real m_real_value; //!< Valeur de type réel
  Real2 m_real2_value; //!< Valeur de type vecteur de 2 réels
  Real3 m_real3_value; //!< Valeur de type vecteur de 3 réel
  Real2x2 m_real2x2_value; //!< Valeur de type matrice 2x2 de réels
  Real3x3 m_real3x3_value; //!< Valeur de type matrice 3x3 de réels
  Int32 m_int32_value; //!< Valeur de type entier 32 bits
  Int64 m_int64_value; //!< Valeur de type eniter 64 bits
  bool m_bool_value; //!< Valeur de type entier booléenne
  String m_string_value; //!< Valeur de type chaîne de caractère.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
