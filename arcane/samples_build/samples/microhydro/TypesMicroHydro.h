// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TypesMicroHydro.h                                           (C) 2000-2022 */
/*                                                                           */
/* Types de MicroHydro (dans axl)                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef TYPESMICROHYDRO_H
#define TYPESMICROHYDRO_H

#include <arcane/ItemGroup.h>
#include "eos/IEquationOfState.h"

struct TypesMicroHydro
{
  enum eBoundaryCondition
  {
    VelocityX, //!< Vitesse X fixée
    VelocityY, //!< Vitesse Y fixée
    VelocityZ, //!< Vitesse Z fixée
    Unknown //!< Type inconnu
  };
};

#endif

