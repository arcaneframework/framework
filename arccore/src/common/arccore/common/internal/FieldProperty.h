// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FieldProperty.h                                             (C) 2000-2026 */
/*                                                                           */
/* Gestion des propriétés comme champ de classes.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_FIELDPROPERTY_H
#define ARCCORE_COMMON_INTERNAL_FIELDPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"
#include "arccore/base/internal/ConvertInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PropertyImpl
{
 public:

  template <typename DataType>
  class FieldProperty
  {
   public:

    explicit FieldProperty(const DataType& default_value)
    : m_value(default_value)
    , m_default_value(default_value)
    {}
    FieldProperty()
    : FieldProperty(DataType())
    {}
    FieldProperty& operator=(const DataType& v)
    {
      setValue(v);
      return (*this);
    }
    explicit(false) operator DataType() const { return m_value; }

   public:

    void setValue(const DataType& v)
    {
      if (m_validator) {
        DataType copy(v);
        m_validator(copy);
        m_value = copy;
      }
      else
        m_value = v;
      m_has_value = true;
    }
    DataType value() const { return m_value; }
    bool isValueSet() const { return m_has_value; }
    void setValidator(std::function<void(DataType&)>&& func) { m_validator = func; }

   private:

    DataType m_value;
    DataType m_default_value;
    bool m_has_value = false;
    std::function<void(DataType&)> m_validator;
  };

  class Int32Value
  {
   public:

    explicit Int32Value(Int32 v)
    : value(v)
    {}
    explicit(false) operator Int32() const { return value; }

   public:

    Int32Value minValue(Int32 x) const
    {
      return Int32Value(std::max(value, x));
    }
    Int32Value maxValue(Int32 x) const
    {
      return Int32Value(std::min(value, x));
    }

   public:

    Int32 value;
  };

  static Int32Value getInt32(const String& str_value, Int32 default_value)
  {
    Int32 v = default_value;
    if (!str_value.null()) {
      bool is_bad = Convert::Impl::StringViewToIntegral::getValue(v, str_value);
      if (is_bad)
        v = default_value;
    }
    return Int32Value(v);
  }
  static void checkSet(FieldProperty<bool>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    bool v = false;
    bool is_bad = Convert::Impl::StringViewToIntegral::getValue(v, str_value);
    if (!is_bad)
      p.setValue(v);
  }
  static void checkSet(FieldProperty<Int32>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    Int32 v = 0;
    bool is_bad = Convert::Impl::StringViewToIntegral::getValue(v, str_value);
    if (!is_bad)
      p.setValue(v);
  }
  static void checkSet(FieldProperty<StringList>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    StringList s;
    s.add(str_value);
    p.setValue(s);
  }
  static void checkSet(FieldProperty<StringList>& p, const StringList& str_values)
  {
    if (p.isValueSet())
      return;
    p.setValue(str_values);
  }
  static void checkSet(FieldProperty<String>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    p.setValue(str_value);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class PropertyKeyValues
  {
   private:

    class NameValuePair
    {
     public:

      NameValuePair(const String& n, const String& v)
      : name(n)
      , value(v)
      {}
      String name;
      String value;
    };

   public:

    /*!
   * \brief Récupère la valeur d'une option.
   *
   * L'ordre de récupération est le suivant:
   * - si \a param_name est non nul, regarde s'il existe une valeur
   * dans \a m_values associée à ce paramètre. Si oui, on retourne cette
   * valeur.
   * - pour chaque nom \a x de \a env_values, regarde si une variable
   * d'environnement \a x existe et retourne sa valeur si c'est le cas.
   * - si aucune des méthodes précédente n'a fonctionné, retourne
   * la valeur \a default_value.
   */
    String getValue(const UniqueArray<String>& env_values, const String& param_name,
                    const String& default_value)
    {
      if (!param_name.null()) {
        String v = _searchParam(param_name);
        if (!v.null())
          return v;
      }
      for (const auto& x : env_values) {
        String ev = platform::getEnvironmentVariable(x);
        if (!ev.null())
          return ev;
      }
      return default_value;
    }
    void add(const String& name, const String& value)
    {
      m_values.add(NameValuePair(name, value));
    }

   private:

    UniqueArray<NameValuePair> m_values;

   private:

    String _searchParam(const String& param_name)
    {
      String v;
      // Une option peut être présente plusieurs fois. Prend la dernière.
      for (const auto& x : m_values) {
        if (x.name == param_name)
          v = x.value;
      }
      return v;
    }
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
