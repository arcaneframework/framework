// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPhysicalUnitSystemService.h                                (C) 2000-2025 */
/*                                                                           */
/* Interface of a unit system.                                               */
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
 * \brief Interface of a unit system.
 */
class ARCANE_CORE_EXPORT IPhysicalUnitSystem
{
 public:

  virtual ~IPhysicalUnitSystem() = default; //!< Releases resources.

 public:

  /*!
   * \brief Creates a converter between two units.
   * The caller must destroy the returned converter.
   * The units \a from and \a to must have been created by this
   * unit system.
   */
  virtual IPhysicalUnitConverter* createConverter(IPhysicalUnit* from, IPhysicalUnit* to) = 0;

  /*!
   * \brief Creates a converter between two units.
   * The caller must destroy the returned converter.
   */
  virtual IPhysicalUnitConverter* createConverter(const String& from, const String& to) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
