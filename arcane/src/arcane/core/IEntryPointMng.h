// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IEntryPointMng.h                                            (C) 2000-2023 */
/*                                                                           */
/* Interface du gestionnaire des points d'entrée.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IENTRYPOINTMNG_H
#define ARCANE_CORE_IENTRYPOINTMNG_H
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
 * \internal
 * \brief Interface du gestionnaire de point d'entrée.
 */
class IEntryPointMng
{
 public:

  virtual ~IEntryPointMng() = default; //!< Libère les ressources.

 public:

  //! Ajoute un point d'entrée au gestionnaire
  virtual void addEntryPoint(IEntryPoint*) =0;

  /*!
   * \brief Point d'entrée de nom \a s.
   *
   * Retourne \a nullptr si le point d'entrée n'est pas trouvé
   */
  virtual IEntryPoint* findEntryPoint(const String& s) =0;

  /*!
   * \brief Point d'entrée de nom \a s du module de nom \a module_name.
   *
   * Retourne \a nullptr si le point d'entrée n'est pas trouvé
   */
  virtual IEntryPoint* findEntryPoint(const String& module_name,const String& s) =0;

  //! Affiche dans \o la liste des points d'entrée du gestionnaire
  virtual void dumpList(std::ostream& o) =0;

  //! Liste des points d'entrées
  virtual EntryPointCollection entryPoints() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

