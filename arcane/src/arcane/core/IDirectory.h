﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectory.h                                                (C) 2000-2018 */
/*                                                                           */
/* Gestion d'un répertoire.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDIRECTORY_H
#define ARCANE_IDIRECTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une classe gérant un répertoire.
 */
class ARCANE_CORE_EXPORT IDirectory
{
 public:

  virtual ~IDirectory() {} //!< Libère les ressources

 public:
	
  /*!
   * \brief Créé le répertoire.
   * \retval true en cas d'échec,
   * \retval false en cas de succès ou si le répertoire existe déjà.
   */
  virtual bool createDirectory() const =0;
	
  //! Retourne le chemin du répertoire
  virtual String path() const =0;

  //! Retourne le chemin complet du fichier \a file_name dans le répertoire
  virtual String file(const String& file_name) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

