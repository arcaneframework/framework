// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Directory.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Gestion d'un répertoire.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DIRECTORY_H
#define ARCANE_CORE_DIRECTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/IDirectory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 * \brief Classe gérant un répertoire.
 */
class ARCANE_CORE_EXPORT Directory
: public IDirectory
{
 public:

  Directory() = default;
  explicit Directory(const String& path);
  Directory(const Directory& directory);
  Directory(const IDirectory& directory, const String& sub_path);
  Directory(const IDirectory& directory);

 public:

  Directory& operator=(const IDirectory& from);
  Directory& operator=(const Directory& from);

 public:

  bool createDirectory() const override;
  String path() const override;
  String file(const String& file_name) const override;

 private:

  String m_directory_path;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

