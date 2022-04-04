// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SmallVariant.h                                              (C) 2000-2017 */
/*                                                                           */
/* Type de base polymorphe.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_SMALLVARIANT_H
#define ARCANE_DATATYPE_SMALLVARIANT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Convert.h"
#include <climits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant un type polymorphe.
 */
class SmallVariant
{
 private:

 public:

  enum eType
  {
    TUnknown = 0,
    TReal= 1,
    TInt32 = 2,
    TInt64 = 3,
    TBool = 4,
    TString = 5
  };

 public:

  static inline int convertFromReal(int,Real v)
    {
      if (v>Convert::toReal(INT_MAX))
        return INT_MAX;
      if (v<Convert::toReal(INT_MIN))
        return INT_MIN;
      return (int)(Convert::toDouble(v));
    }
  static inline unsigned int convertFromReal(unsigned int,Real v)
    {
      if (v>Convert::toReal(UINT_MAX))
        return UINT_MAX;
      if (v<0.0)
        return 0;
      return (unsigned int)(Convert::toDouble(v));
    }
  static inline long convertFromReal(long,Real v)
    {
      if (v>Convert::toReal(LONG_MAX))
        return LONG_MAX;
      if (v<Convert::toReal(LONG_MIN))
        return LONG_MIN;
      return (long)(Convert::toDouble(v));
    }
  static inline unsigned long convertFromReal(unsigned long,Real v)
    {
      if (v>Convert::toReal(ULONG_MAX))
        return ULONG_MAX;
      if (v<0.0)
        return 0;
      return (unsigned long)(Convert::toDouble(v));
    }
  static inline long long convertFromReal(long long,Real v)
    {
      if (v>Convert::toReal(LLONG_MAX))
        return LLONG_MAX;
      if (v<Convert::toReal(LLONG_MIN))
        return LLONG_MIN;
      return (long long)(Convert::toDouble(v));
    }
  static inline unsigned long long convertFromReal(unsigned long long,Real v)
    {
      if (v>Convert::toReal(ULLONG_MAX))
        return ULLONG_MAX;
      if (v<0.0)
        return 0;
      return (unsigned long long)(Convert::toDouble(v));
    }

 public:

  SmallVariant()
  : m_real_value(0.), m_int32_value(0), m_int64_value(0),
    m_bool_value(false), m_sticky_type(TUnknown) {}
  SmallVariant(Real v)
  : m_real_value(v), m_int32_value(0), m_int64_value(0),
    m_bool_value(false), m_sticky_type(TReal) {}
  SmallVariant(Int32 v)
  : m_real_value(0.), m_int32_value(v), m_int64_value(0),
    m_bool_value(false), m_sticky_type(TInt32) {}
  SmallVariant(Int64 v)
  : m_real_value(0.), m_int32_value(0), m_int64_value(v),
    m_bool_value(false), m_sticky_type(TInt64) {}
  SmallVariant(bool v)
  : m_real_value(0.), m_int32_value(0), m_int64_value(v),
    m_bool_value(v), m_sticky_type(TBool) {}
  SmallVariant(const String& v)
  : m_real_value(0.), m_int32_value(0), m_int64_value(0),
    m_bool_value(false), m_string_value(v), m_sticky_type(TString) {}

  void setValue(Real v) { m_real_value = v; m_sticky_type = TReal; }
  void setValue(Int32 v) { m_int32_value = v; m_sticky_type = TInt32; }
  void setValue(Int64 v) { m_int64_value = v; m_sticky_type = TInt64; }
  void setValue(const String& v) { m_string_value = v; m_sticky_type = TString; }
  void setValue(bool v) { m_bool_value = v; m_sticky_type = TBool; }

  void setValueAll(Real v)
  {
    m_sticky_type = TReal;
    m_real_value = v;
    m_int32_value = convertFromReal(m_int32_value,v);
    m_int64_value = convertFromReal(m_int64_value,v);
    m_bool_value = v!=0.;
    m_string_value = String::fromNumber(m_real_value);
  }
  void setValueAll(Int32 v)
  {
    m_sticky_type = TInt32;
    m_real_value = Convert::toReal(v);
    m_int32_value = v;
    m_int64_value = (Int64)(v);
    m_bool_value = v!=0;
    m_string_value = String::fromNumber(m_int32_value);
  }
  void setValueAll(Int64 v)
  {
    m_sticky_type = TInt64;
    m_real_value = Convert::toReal(v);
    m_int32_value = (Int32)(v);
    m_int64_value = v;
    m_bool_value = v!=0;
    m_string_value = String::fromNumber(m_int64_value);
  }
  void setValueAll(bool v)
  {
    m_sticky_type = TBool;
    m_real_value = Convert::toReal(v);
    m_int32_value = v ? 1 : 0;
    m_int64_value = v ? 1 : 0;
    m_bool_value = v;
    m_string_value = String::fromNumber(m_int64_value);
  }

  void value(bool& v) const { v = m_bool_value; }
  void value(Real& v) const { v = m_real_value; }
  void value(Int32& v) const { v = m_int32_value; }
  void value(Int64& v) const { v = m_int64_value; }
  void value(String& v) const { v = m_string_value; }

  bool asBool() const { return m_bool_value; }
  Real asReal() const { return m_real_value; }
  Integer asInteger() const;
  Int32 asInt32() const { return m_int32_value; }
  Int64 asInt64() const { return m_int64_value; }
  const String& asString() const { return m_string_value; }
  eType type() const { return m_sticky_type; }

 private:

  Real m_real_value;     //!< Valeur de type réelle
  Int32 m_int32_value;  //!< Valeur de type entier
  Int64 m_int64_value;  //!< Valeur de type entier naturel
  bool m_bool_value;  //!< Valeur de type entier booléenne
  String m_string_value;   //!< Valeur de type chaîne de caractère.
  eType m_sticky_type; //!< Type garanti valide de la valeur.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type>
class VariantGetterT
{
 public:
  VariantGetterT() {}
  virtual ~VariantGetterT() {}
  static Type asType(const SmallVariant& v)
  { Type t; v.value(t); return t; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

