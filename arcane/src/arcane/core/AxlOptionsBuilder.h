// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AxlOptionsBuilder.h                                         (C) 2000-2023 */
/*                                                                           */
/* Classes for dynamically creating data set options.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_AXLOPTIONSBUILDER_H
#define ARCANE_CORE_AXLOPTIONSBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/ArcaneTypes.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AxlOptionsBuilder
{
class OneOption;
class OneOptionImpl;
class DocumentXmlWriter;
class DocumentJSONWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data set option list.
 */
class ARCANE_CORE_EXPORT OptionList
{
  friend OneOption;
  friend OneOptionImpl;
  friend DocumentXmlWriter;
  friend DocumentJSONWriter;

 public:

  //! Constructs an empty set of options.
  OptionList();
  //! Constructs a list of options
  explicit OptionList(const std::initializer_list<OneOption>& options);

 public:

  OptionList& add(const String& name, const OptionList& option);
  OptionList& add(const OneOption& opt);
  OptionList& add(const std::initializer_list<OneOption>& options);

 public:

  OptionList clone() const;

 private:

  std::shared_ptr<OneOptionImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for a dynamic option.
 */
class ARCANE_CORE_EXPORT OneOption
{
  friend class OptionList;
  friend class OneOptionImpl;
  friend DocumentXmlWriter;
  friend DocumentJSONWriter;

 protected:

  enum class Type
  {
    CO_Simple,
    CO_Enumeration,
    CO_Extended,
    CO_Complex,
    CO_ServiceInstance
  };

 public:

  OneOption() = default;

 protected:

  OneOption(Type type, const String& name, const String& value)
  : m_type(type)
  , m_name(name)
  , m_value(value)
  {}
  OneOption(Type type, const String& name, const OptionList& option);

 protected:

  Type m_type = Type::CO_Simple;
  String m_name; //! Option name
  String m_value; //! Option value (if CO_Simple option)
  String m_service_name; //!< Service name (if CO_ServiceInstance type option)
  String m_function_name; //<! Function name (ICaseFunction)
  std::shared_ptr<OneOptionImpl> m_sub_option;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data set 'Simple' option.
 */
class ARCANE_CORE_EXPORT Simple
: public OneOption
{
 public:

  Simple(const String& name, Int32 value)
  : OneOption(Type::CO_Simple, name, String::fromNumber(value))
  {}

  Simple(const String& name, Int64 value)
  : OneOption(Type::CO_Simple, name, String::fromNumber(value))
  {
  }

  Simple(const String& name, Real value)
  : OneOption(Type::CO_Simple, name, String::fromNumber(value))
  {
  }

  Simple(const String& name, const String& value)
  : OneOption(Type::CO_Simple, name, value)
  {
  }

 public:

  Simple& addFunction(const String& func_name)
  {
    m_function_name = func_name;
    return (*this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data set 'Enumeration' option.
 */
class ARCANE_CORE_EXPORT Enumeration
: public OneOption
{
 public:

  Enumeration(const String& name, const String& value)
  : OneOption(Type::CO_Enumeration, name, value)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data set 'Extended' option.
 */
class ARCANE_CORE_EXPORT Extended
: public OneOption
{
 public:

  Extended(const String& name, const String& value)
  : OneOption(Type::CO_Extended, name, value)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data set 'Complex' option.
 */
class ARCANE_CORE_EXPORT Complex
: public OneOption
{
 public:

  Complex(const String& name, const std::initializer_list<OneOption>& options);
  Complex(const String& name, const OptionList& option);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data set 'ServiceInstance' option.
 */
class ARCANE_CORE_EXPORT ServiceInstance
: public OneOption
{
 public:

  ServiceInstance(const String& option_name, const String& service_name,
                  const std::initializer_list<OneOption>& options);
  ServiceInstance(const String& option_name, const String& service_name,
                  const OptionList& options);
  ServiceInstance(const String& option_name, const String& service_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
* \brief Data set 'ServiceInstance' option.
 */
class ARCANE_CORE_EXPORT Document
{
  friend DocumentXmlWriter;
  friend DocumentJSONWriter;

 public:

  Document(const String& lang, const OptionList& options)
  : m_language(lang)
  , m_options(options)
  {}

 public:

  const String& language() const { return m_language; }

 private:

  String m_language;
  OptionList m_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::AxlOptionsBuilder

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
