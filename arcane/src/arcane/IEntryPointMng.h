// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IEntryPointMng.h                                            (C) 2000-2018 */
/*                                                                           */
/* Interface du gestionnaire des points d'entrée.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IENTRYPOINTMNG_H
#define ARCANE_IENTRYPOINTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IEntryPoint;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface du gestionnaire de point d'entrée.
 */
class IEntryPointMng
{
 public:

  virtual ~IEntryPointMng() {} //!< Libère les ressources.

 public:

  //! Ajoute un point d'entrée au gestionnaire
  virtual void addEntryPoint(IEntryPoint*) =0;

  //! Point d'entrée de nom \a s
  virtual IEntryPoint* findEntryPoint(const String& s) =0;

  //! Point d'entrée de nom \a s du module de nom \a module_name
  virtual IEntryPoint* findEntryPoint(const String& module_name,const String& s) =0;

  //! Affiche la liste des points d'entrée du gestionnaire
  virtual void dumpList(std::ostream&) =0;

  //! Liste des points d'entrées
  virtual EntryPointCollection entryPoints() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

