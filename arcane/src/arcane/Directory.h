// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Directory.h                                                 (C) 2000-2016 */
/*                                                                           */
/* Gestion d'un répertoire.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DIRECTORY_H
#define ARCANE_DIRECTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDirectory.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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

  Directory();
  explicit Directory(const String& path);
  Directory(const Directory& directory);
  Directory(const IDirectory& directory,const String& sub_path);
  Directory(const IDirectory& directory);
  virtual ~Directory(); //!< Libère les ressources

  public:
  
  const Directory& operator=(const IDirectory& from);
  const Directory& operator=(const Directory& from);

 public:
	
  virtual bool createDirectory() const;
  virtual String path() const;
  virtual String file(const String& file_name) const;

 private:

  String m_directory_path;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

