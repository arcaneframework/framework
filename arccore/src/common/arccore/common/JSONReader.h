// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONReader.h                                                (C) 2000-2025 */
/*                                                                           */
/* JSON format reader.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_JSONREADER_H
#define ARCCORE_COMMON_JSONREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"
#include "arccore/common/CommonGlobal.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Represents a JSON value.
 *
 * Instances of this class are only valid as long as the associated document
 * exists.
 *
 * \warning Internal API. Do not use outside of Arcane.
 */
class ARCCORE_COMMON_EXPORT JSONValue
{
  class Impl;
  friend JSONWrapperUtils;
  friend JSONKeyValue;

 private:

  explicit JSONValue(Impl* p)
  : m_p(p)
  {}

 public:

  JSONValue()
  : m_p(nullptr)
  {}

 public:

  //! True if the node is null
  bool null() const { return !m_p; }
  bool operator!() const { return null(); }

 public:

  ARCCORE_DEPRECATED_REASON("Y2023: Use valueAsStringView() or value() instead")
  StringView valueAsString() const;

  //! Value in String format. The returned string is null if 'null()' is true.
  String value() const;
  /*!
   * \brief Value in StringView format.
   * The string is empty if 'null()' is true.
   * \note If you want to distinguish between a null value and an empty string,
   * you must use value().
   */
  StringView valueAsStringView() const;
  //! Value in Real format. Returns 0.0 if 'null()' is true.
  Real valueAsReal() const;
  //! Value in Int64 format. Returns 0 if 'null()' is true.
  Int64 valueAsInt64() const;
  //! Value in Int64 format. Returns 0 if 'null()' is true.
  Int32 valueAsInt32() const;
  //! Value in boolean format. Returns false if 'null()' is true.
  bool valueAsBool() const;
  JSONValueList valueAsArray() const;

 public:

  JSONKeyValue keyValueChild(StringView name) const;
  //! Child value with name \a name. Returns a null value if not found.
  JSONValue child(StringView name) const;
  //! Child value with name \a name. Throws an exception if not found.
  JSONValue expectedChild(StringView name) const;
  // List of child objects of this object. The instance must be an object
  JSONValueList children() const;
  JSONKeyValueList keyValueChildren() const;

 public:

  bool isArray() const;
  bool isObject() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Represents a (key,value) pair of JSON.
 *
 * Instances of this class are only valid as long as the associated document
 * exists.
 *
 * \warning Internal API. Do not use outside of Arcane.
 */
class ARCCORE_COMMON_EXPORT JSONKeyValue
{
  class Impl;
  friend JSONWrapperUtils;

 private:

  explicit JSONKeyValue(Impl* p)
  : m_p(p)
  {}

 public:

  JSONKeyValue()
  : m_p(nullptr)
  {}

 public:

  //! True if the node is null
  bool null() const { return !m_p; }
  bool operator!() const { return null(); }

 public:

  StringView name() const;
  JSONValue value() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief List of (key,value) pairs of a JSON document.
 *
 * Instances of this class are only valid as long as the associated document
 * exists.
 *
 * \warning Internal API. Do not use outside of Arcane.
 */
class ARCCORE_COMMON_EXPORT JSONKeyValueList
{
  typedef std::vector<JSONKeyValue> ContainerType;

 public:

  typedef ContainerType::const_iterator const_iterator;
  typedef ContainerType::iterator iterator;

 public:

  void add(JSONKeyValue v)
  {
    m_values.push_back(v);
  }
  const_iterator begin() const { return m_values.begin(); }
  const_iterator end() const { return m_values.end(); }

 private:

  std::vector<JSONKeyValue> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief List of values of a JSON document.
 *
 * Instances of this class are only valid as long as the associated document
 * exists.
 *
 * \warning Internal API. Do not use outside of Arcane.
 */
class ARCCORE_COMMON_EXPORT JSONValueList
{
  typedef std::vector<JSONValue> ContainerType;

 public:

  typedef ContainerType::const_iterator const_iterator;
  typedef ContainerType::iterator iterator;

 public:

  void add(JSONValue v)
  {
    m_values.push_back(v);
  }
  const_iterator begin() const { return m_values.begin(); }
  const_iterator end() const { return m_values.end(); }

 private:

  std::vector<JSONValue> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Management of a JSON document.
 *
 * \warning Internal API. Do not use outside of Arcane.
 */
class ARCCORE_COMMON_EXPORT JSONDocument
{
  class Impl;

 public:

  JSONDocument();
  ~JSONDocument();

 public:

  //! Reads the file in UTF-8 format.
  void parse(Span<const Byte> bytes);
  //! Reads the file in UTF-8 format.
  void parse(Span<const std::byte> bytes);
  //! Reads the file in UTF-8 format.
  void parse(Span<const Byte> bytes, StringView file_name);
  //! Reads the file in UTF-8 format.
  void parse(Span<const std::byte> bytes, StringView file_name);
  //! Root element
  JSONValue root() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
