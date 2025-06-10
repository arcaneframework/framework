// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPhysicalUnitSystemService.h                                (C) 2000-2025 */
/*                                                                           */
/* Interface d'un système d'unité.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPHYSICALUNITSYSTEM_H
#define ARCANE_CORE_IPHYSICALUNITSYSTEM_H
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
 * \internal
 * \brief Interface d'un système d'unité.
 */
class ARCANE_CORE_EXPORT IPhysicalUnitSystem
{
 public:

  virtual ~IPhysicalUnitSystem() = default; //!< Libère les ressources.

 public:

  /*!
   * \brief Créé un convertisseur entre deux unités.
   * L'appelant doit détruire le convertisseur retourné.
   * Les unités \a from et \a to doivent avoir été créées par ce
   * système d'unité.
   */
  virtual IPhysicalUnitConverter* createConverter(IPhysicalUnit* from, IPhysicalUnit* to) = 0;

  /*!
   * \brief Créé un convertisseur entre deux unités.
   * L'appelant doit détruire le convertisseur retourné.
   */
  virtual IPhysicalUnitConverter* createConverter(const String& from, const String& to) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
