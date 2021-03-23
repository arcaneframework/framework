// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FileContent.h                                               (C) 2000-2019 */
/*                                                                           */
/* Contenu d'un fichier.                                                     */
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
 * \brief Description et contenu d'un fichier.
 */
class ARCANE_UTILS_EXPORT FileContent
{
 public:
  static const Int32 CURRENT_VERSION = 1;
 public:

  //! Créé un contenu vide.
  FileContent() : m_version(CURRENT_VERSION) {}
  FileContent(Span<const Byte> abytes,Int32 version,const String& compression)
  : m_bytes(abytes), m_version(version), m_compression(compression){}

 public:

  //! Contenu du fichier
  Span<const Byte> bytes() const;
  //! Version du contenu
  Int32 version() const { return m_version; }
  //! Algorithme de compression utilisé.
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

