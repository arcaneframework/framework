// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPhysicalUnitSystemService.h                                (C) 2000-2025 */
/*                                                                           */
/* Interface of a service managing a unit system.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPHYSICALUNITSYSTEMSERVICE_H
#define ARCANE_CORE_IPHYSICALUNITSYSTEMSERVICE_H
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
 * \brief Interface of a service managing a unit system.
 */
class ARCANE_CORE_EXPORT IPhysicalUnitSystemService
{
 public:

  virtual ~IPhysicalUnitSystemService() = default; //!< Releases resources.

 public:

  virtual void build() = 0;

 public:

  /*!
   * \brief Creates a unit system for the International System SI.
   */
  virtual IPhysicalUnitSystem* createStandardUnitSystem() = 0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
