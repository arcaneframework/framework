// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRessourceMng.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface of a resource manager.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IRESSOURCEMNG_H
#define ARCANE_CORE_IRESSOURCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a resource manager.
 *
 */
class ARCANE_CORE_EXPORT IRessourceMng
{
  // TODO: delete this class which is no longer useful.
  // It is possible to create an instance directly
  // of IXmlDocumentHolder

 public:

  //! Creation of a default history manager.
  static IRessourceMng* createDefault(IApplication*);

 public:

  virtual ~IRessourceMng() = default; //!< Frees the resources

 public:

  /*!
   * \brief Creates an XML document node.
   *
   * Creates and returns an XML document using a default implementation.
   * The destruction of this document invalidates all nodes that depend on it.
   */
  virtual IXmlDocumentHolder* createXmlDocument() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
