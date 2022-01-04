// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleProperty.h                                            (C) 2000-2018 */
/*                                                                           */
/* Propriétés d'un module.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MODULEPROPERTY_H
#define ARCANE_MODULEPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Propriétés de création d'un module.
 *
 * Cette classe est utilisé dans les macros d'enregistrement des modules
 * et peut donc être instantiée en tant que variable globale avant d'entrer
 * dans le main() du code. Elle ne doit donc contenir que des champs de type
 * Plain Object Data (POD).
 *
 * En général, lest instances de cette classes sont utilisés lors
 * de l'enregistrement d'un service via la macro ARCANE_REGISTER_MODULES().
 */
class ARCANE_CORE_EXPORT ModuleProperty
{
 public:

  /*!
   * \brief Construit une instance pour un module de nom \a aname.
   */
  ModuleProperty(const char* aname,bool is_autoload) ARCANE_NOEXCEPT
  : m_name(aname), m_is_autoload(is_autoload)
  {
  }

  /*!
   * \brief Construit une instance pour un module de nom \a aname.
   */
  explicit ModuleProperty(const char* aname) ARCANE_NOEXCEPT
  : m_name(aname), m_is_autoload(false)
  {
  }

 public:

  //! Nom du module.
  const char* name() const { return m_name; }
  
  //! Indique si le module est automatiquement chargé.
  bool isAutoload() const { return m_is_autoload; }

 private:

  const char* m_name;
  bool m_is_autoload;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

