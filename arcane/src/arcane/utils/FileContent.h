// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FileContent.h                                               (C) 2000-2019 */
/*                                                                           */
/* File content.                                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FILECONTENT_H
#define ARCANE_UTILS_FILECONTENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Description and content of a file.
 */
class ARCANE_UTILS_EXPORT FileContent
{
 public:

  static const Int32 CURRENT_VERSION = 1;

 public:

  //! Creates empty content.
  FileContent()
  : m_version(CURRENT_VERSION)
  {}
  FileContent(Span<const Byte> abytes, Int32 version, const String& compression)
  : m_bytes(abytes)
  , m_version(version)
  , m_compression(compression)
  {}

 public:

  //! File content
  Span<const Byte> bytes() const;
  //! Content version
  Int32 version() const { return m_version; }
  //! Compression algorithm used.
  const String& compression() const { return m_compression; }

 private:

  UniqueArray<Byte> m_bytes;
  Int32 m_version;
  String m_compression;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
