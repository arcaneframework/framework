// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneException.h                                           (C) 2000-2025 */
/*                                                                           */
/* Exceptions thrown by Arcane.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ARCANEEXCEPTION_H
#define ARCANE_CORE_ARCANEEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"
#include "arcane/utils/String.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception for an invalid identifier.
 *
 * This exception is thrown whenever an invalid identifier
 * is used in the architecture.

 The following rules must be respected for an identifier to be
 valid:
 
 \arg it must contain at least one character.
 \arg it must start with an alphabetic character (a-zA-Z),
 \arg it must be followed by a sequence of alphabetic characters, digits,
 or the underscore character '_'.
 */
class ARCANE_CORE_EXPORT BadIDException
: public Exception
{
 public:

  /*!
   * Constructs an exception related to the manager \a m, originating from the function
   * \a where and with the invalid name \a invalid_name.
   */
  BadIDException(const String& where, const String& invalid_name);
  ~BadIDException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  String m_invalid_name; //!< Invalid identifier.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception for an invalid entity ID.
 *
 * This exception is thrown whenever an entity ID (whether local or global)
 * is invalid.
 */
class ARCANE_CORE_EXPORT BadItemIdException
: public Exception
{
 public:

  /*!
   * \brief Constructs an exception.
   
   Constructs an exception related to the message manager \a m,
   originating from the function \a where and with the invalid ID \a id.
   */
  BadItemIdException(const String& where, Integer bad_id);
  ~BadItemIdException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  Integer m_bad_id; //!< Invalid ID.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception when an internal error occurs.
 */
class ARCANE_CORE_EXPORT InternalErrorException
: public Exception
{
 public:

  InternalErrorException(const String& where, const String& why);
  InternalErrorException(const TraceInfo& where, const String& why);
  InternalErrorException(const InternalErrorException& ex) ARCANE_NOEXCEPT;
  ~InternalErrorException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  String m_why;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception for an invalid variable kind/type.
 *
 * This exception is thrown when attempting to reference a variable
 * that already exists in another module with a different kind or type.
 */
class ARCANE_CORE_EXPORT BadVariableKindTypeException
: public Exception
{
 public:

  BadVariableKindTypeException(const TraceInfo& where, IVariable* valid_var,
                               eItemKind kind, eDataType datatype, int dimension);
  ~BadVariableKindTypeException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  IVariable* m_valid_var;
  eItemKind m_item_kind;
  eDataType m_data_type;
  int m_dimension;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception for an invalid partial variable item group name.
 *
 * This exception is thrown when attempting to reference a partial variable
 * that already exists in another module with a different item group name.
 */
class ARCANE_CORE_EXPORT BadPartialVariableItemGroupNameException
: public Exception
{
 public:

  BadPartialVariableItemGroupNameException(const TraceInfo& where, IVariable* valid_var,
                                           const String& item_group_name);
  ~BadPartialVariableItemGroupNameException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  IVariable* m_valid_var;
  String m_item_group_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception when a mesh entity is not of a known type.
 */
class ARCANE_CORE_EXPORT UnknownItemTypeException
: public Exception
{
 public:

  UnknownItemTypeException(const String& where, Integer nb_node, Integer item_id);
  UnknownItemTypeException(const UnknownItemTypeException& ex) ARCANE_NOEXCEPT;
  ~UnknownItemTypeException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  Integer m_nb_node;
  Integer m_item_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception when trying to dereference a null pointer.
 */
class ARCANE_CORE_EXPORT BadReferenceException
: public Exception
{
 public:

  explicit BadReferenceException(const String& where);
  ~BadReferenceException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception in a reader or writer.
 */
class ARCANE_CORE_EXPORT ReaderWriterException
: public Exception
{
 public:

  ReaderWriterException(const String& where, const String& message);
  ReaderWriterException(const TraceInfo& where, const String& message);
  ReaderWriterException(const ReaderWriterException& ex) ARCANE_NOEXCEPT;
  ~ReaderWriterException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exception in an assertion.
 */
class ARCANE_CORE_EXPORT AssertionException
: public Exception
{
 public:

  /*!
   * Constructs an exception originating from the function \a where.
   */
  explicit AssertionException(const TraceInfo& where);

  /*!
   * Constructs an exception originating from the function \a where.
   * The expected value in the assertion was \a expected, the obtained result was \a actual.
   */
  AssertionException(const TraceInfo& where, const String& expected, const String& actual);

 public:

  void explain(std::ostream& m) const override;

  //! File of the exception
  const char* file() const { return m_file; }

  //! Line of the exception
  int line() const { return m_line; }

 public:

  using Exception::message;
  using Exception::where;

 private:

  const char* m_file;
  int m_line;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
