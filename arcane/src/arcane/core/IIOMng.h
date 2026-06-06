// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIOMng.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Interface of the input-output manager.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IIOMNG_H
#define ARCANE_CORE_IIOMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup IO
 * \brief Interface of the input/output manager.
 *
 * \todo input/output manager allowing encapsulation
 * of file management in parallel.
 */
class ARCANE_CORE_EXPORT IIOMng
{
 public:

  virtual ~IIOMng() = default; //!< Frees resources

 public:

  /*!
   * \brief Reads and parses the XML file \a filename.
   *
   * In case of an error, returns 0.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   * If a schema name is specified, the consistency
   * of the file relative to the schema is checked.
   */
  virtual IXmlDocumentHolder*
  parseXmlFile(const String& filename, const String& schemaname = String{}) = 0;

  /*!
   * \brief Reads and parses the XML file \a filename.
   *
   * In case of an error, returns 0.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   * The consistency of the file relative to the schema is checked; 
   * The schema name is provided only for error message processing.
   */
  virtual IXmlDocumentHolder* parseXmlFile(const String& filename,
                                           const String& schemaname,
                                           ConstArrayView<Byte> schema_data) = 0;

  /*!
   * \brief Reads and parses the XML file contained in the buffer \a buffer.
   *
   * In case of an error, returns 0.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   * The argument \a name associates a name with the memory area that is
   * used for displaying error messages.
   */
  virtual IXmlDocumentHolder* parseXmlBuffer(Span<const Byte> buffer, const String& name) = 0;

  /*!
   * \brief Reads and parses the XML file contained in the buffer \a buffer.
   *
   * In case of an error, returns 0.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   * The argument \a name associates a name with the memory area that is
   * used for displaying error messages.
   */
  virtual IXmlDocumentHolder* parseXmlBuffer(Span<const std::byte> buffer, const String& name) = 0;

  /*!
   * \brief Reads and parses the XML file contained in the string \a str.
   *
   * In case of an error, returns 0.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   * The argument \a name associates a name with the memory area that is
   * used for displaying error messages.
   */
  virtual IXmlDocumentHolder* parseXmlString(const String& str, const String& name) = 0;

  /*! \brief Writes the XML tree of the document \a doc to the file filename.
   * \retval true in case of an error,
   * \return false in case of success.
   */
  virtual bool writeXmlFile(IXmlDocumentHolder* doc, const String& filename, const bool indented = false) = 0;

  /*!
   * \brief Collective reading of a file.
   *
   * Collectively reads the file \a filename and returns its
   * content in \a bytes. The file is considered a binary file.
   * Collective reading means that all
   * processors call this operation and will read the same file.
   * The implementation can then optimize disk access by grouping the
   * actual reading on one or more processors and then sending the
   * file content to the others.
   *
   * \retval true in case of an error
   * \retval false if everything is okay.
   */
  virtual bool collectiveRead(const String& filename, ByteArray& bytes) = 0;

  /*!
   * \brief Collective reading of a file.
   *
   * Collectively reads the file \a filename and returns its
   * content in \a bytes. The file is considered a binary file
   * if \a is_binary is true.
   * Collective reading means that all
   * processors call this operation and will read the same file.
   * The implementation can then optimize disk access by grouping the
   * actual reading on one or more processors and then sending the
   * file content to the others.
   *
   * \retval true in case of an error
   * \retval false if everything is okay.
   */
  virtual bool collectiveRead(const String& filename, ByteArray& bytes, bool is_binary) = 0;

  /*!
   * \brief Local reading of a file.
   *
   * Locally reads the file \a filename and returns its
   * content in \a bytes. The file is considered a binary file.
   * This operation is not collective.
   *
   * \retval true in case of an error.
   * \retval false if everything is okay.
   *
   * \warning also returns true if the file is empty.
   * \warning if the ByteUniqueArray must be converted to a String, a terminal 0 must be added beforehand (bytes.add(0))
   */
  virtual bool localRead(const String& filename, ByteArray& bytes) = 0;

  /*!
   * \brief Local reading of a file.
   *
   * Locally reads the file \a filename and returns its
   * content in \a bytes.
   * This operation is not collective.
   *
   * \retval true in case of an error.
   * \retval false if everything is okay.
   *
   * \warning also returns true if the file is empty.
   * \warning if the ByteUniqueArray must be converted to a String, a terminal 0 must be added beforehand (bytes.add(0))
   */
  virtual bool localRead(const String& filename, ByteArray& bytes, bool is_binary) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
