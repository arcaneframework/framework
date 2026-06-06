// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IXmlDocumentHolder.h                                        (C) 2000-2018 */
/*                                                                           */
/* Interface of a DOM document manager.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IXMLDOCUMENTHOLDER_H
#define ARCANE_IXMLDOCUMENTHOLDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNode;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Xml
 * \brief Manager of a DOM document.
 *
 * This class encapsulates the document node of a DOM tree.
 * The destructor of this class releases the DOM tree.
 * The user must be careful not to use
 * a node from this tree after its release.
 */
class ARCANE_CORE_EXPORT IXmlDocumentHolder
{
 public:

  //! Releases resources
  virtual ~IXmlDocumentHolder() {}

 public:

  //! Creates and returns a null document.
  static IXmlDocumentHolder* createNull();

  /*!
   * \brief Loads an XML document.
   *
   * Reads and parses the XML document named \a name whose data is in \a buffer.
   *
   * The returned instance is never null.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   */
  static IXmlDocumentHolder*
  loadFromBuffer(Span<const Byte> buffer, const String& name, ITraceMng* tm);

  /*!
   * \brief Loads an XML document.
   *
   * Reads and parses the XML document named \a name whose data is in \a buffer.
   *
   * The returned instance is never null.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   */
  static IXmlDocumentHolder*
  loadFromBuffer(ByteConstSpan buffer, const String& name, ITraceMng* tm);

  /*!
   * \brief Loads an XML document.
   *
   * Reads and parses the XML document contained in the file \a filename.
   *
   * The returned instance is never null.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   */
  static IXmlDocumentHolder*
  loadFromFile(const String& filename, ITraceMng* tm);

  /*!
   * \brief Loads an XML document.
   *
   * Reads and parses the XML document contained in the file \a filename.
   *
   * The returned instance is never null.
   * The caller owns the returned instance and must
   * destroy it using the delete operator.
   *
   * If \a schema_filename is not null, it indicates the XML file containing the schema
   * used to validate the XML file.
   */
  static IXmlDocumentHolder*
  loadFromFile(const String& filename, const String& schema_filename, ITraceMng* tm);

 public:

  //! Document node. \c This node is null if the document does not exist.
  virtual XmlNode documentNode() = 0;

  //! Clones this document
  virtual IXmlDocumentHolder* clone() = 0;

  //! Saves this document into the array \a bytes.
  virtual void save(ByteArray& bytes) = 0;

  //! Saves this document and returns the string.
  virtual String save() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
