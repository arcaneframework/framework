// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MicroHydroTypes.h                                           (C) 2000-2022 */
/*                                                                           */
/* Types du module d'hydrodynamique.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANETEST_MICROHYDROTYPES_H
#define ARCANETEST_MICROHYDROTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MicroHydro
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MicroHydroTypes
{
 public:

  enum eBoundaryCondition
  {
    VelocityX, //!< Vitesse X fixée
    VelocityY, //!< Vitesse Y fixée
    VelocityZ, //!< Vitesse Z fixée
    Unknown //!< Type inconnu
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IBoundaryCondition
{
 public:

  virtual ~IBoundaryCondition() = default;

 public:

  virtual FaceGroup getSurface() = 0;
  virtual Real getValue() = 0;
  virtual MicroHydroTypes::eBoundaryCondition getType() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace MicroHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
