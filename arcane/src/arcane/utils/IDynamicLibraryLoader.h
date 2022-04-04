﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDynamicLibraryLoader.h                                     (C) 2000-2019 */
/*                                                                           */
/* Interface d'une chargeur dynamique de bibliothèque.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDYNAMICLIBRARYLOADER_H
#define ARCANE_IDYNAMICLIBRARYLOADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une bibliothèque dynamique.
 */
class ARCANE_UTILS_EXPORT IDynamicLibrary
{
 public:
  virtual ~IDynamicLibrary(){} //!< Libère les ressources
 public:
  /*!
   * \brief Ferme la bibliothèque dynamique.
   *
   * Elle ne doit plus être utilisée après fermeture et l'instance peut
   * être détruite via l'opérateur delete.
   */
  virtual void close() =0;

  /*!
   * \brief Retourne l'adresse du symbol de nom \a symbol_name.
   *
   * Si \a is_found n'est pas nul, contient en retour le booléen indiquant
   * si le symbol a été trouvé.
   */
  virtual void* getSymbolAddress(const String& symbol_name,bool* is_found) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal.
 *
 * \brief Interface d'une chargeur dynamique de bibliothèque.
 */
class ARCANE_UTILS_EXPORT IDynamicLibraryLoader
{
 public:
	
  virtual ~IDynamicLibraryLoader(){} //!< Libère les ressources

 public:

  virtual void build() =0;

 public:

  /*!
   * \brief Charge une bibliothèque dynamique.
   *
   * Charge la bibliothèque de nom \a name qui se trouve dans le répertoire
   * \a directory. Retourne un pointeur nul si la bibliothèque ne peut
   * pas être chargée. \a name doit être un nom sans préfixe et sans extension
   * dépendant machine. Par exemple sous linux, si la bibliothèque est
   * libtoto.so, \a name doit valoir \a toto.
   */
  virtual IDynamicLibrary* open(const String& directory,const String& name) =0;

  //! Ferme toutes les bibliothèques ouvertes via \a open()
  virtual void closeLibraries() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

