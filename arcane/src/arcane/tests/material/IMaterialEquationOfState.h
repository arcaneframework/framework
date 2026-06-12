// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCANE_TESTS_IMATERIALEQUATIONOFSTATE_H
#define ARCANE_TESTS_IMATERIALEQUATIONOFSTATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"

#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MeshMaterialVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MaterialEos
{
using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Interface for the equation of state calculation model service.
class IMaterialEquationOfState
{
 public:

  /** Class destructor */
  virtual ~IMaterialEquationOfState() = default;

 public:

  /*!
   *  Initializes the equation of state for the cell group passed as argument
   *  and calculates the sound speed and internal energy.
   */
  virtual void initEOS(IMeshMaterial* mat,
                       const MaterialVariableCellReal& pressure,
                       const MaterialVariableCellReal& density,
                       MaterialVariableCellReal& internal_energy,
                       MaterialVariableCellReal& sound_speed) = 0;
  /*!
   *  Applies the equation of state to the cell group passed as argument
   *  and calculates the sound speed and pressure.
   */
  virtual void applyEOS(IMeshMaterial* mat,
                        const MaterialVariableCellReal& density,
                        const MaterialVariableCellReal& internal_energy,
                        MaterialVariableCellReal& pressure,
                        MaterialVariableCellReal& sound_speed) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace MaterialEos

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
