// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCANE_TESTS_IMATERIALEQUATIONOFSTATE_H
#define ARCANE_TESTS_IMATERIALEQUATIONOFSTATE_H

#include "arcane/ItemTypes.h"
#include "arcane/VariableTypedef.h"

#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MeshMaterialVariableRef.h"


namespace MaterialEos
{
using namespace Arcane;
using namespace Arcane::Materials;

//! Interface du service du modèle de calcul de l'équation d'état.
class IMaterialEquationOfState
{
 public:
  /** Destructeur de la classe */
  virtual ~IMaterialEquationOfState() = default;

 public:
  /*!
   *  Initialise l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et l'énergie interne.
   */
  virtual void initEOS(IMeshMaterial* mat,
                       const MaterialVariableCellReal& pressure,
                       const MaterialVariableCellReal& density,
                       MaterialVariableCellReal& internal_energy,
                       MaterialVariableCellReal& sound_speed
                       ) =0;
  /*!
   *  Applique l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et la pression.
   */
  virtual void applyEOS(IMeshMaterial* mat,
                        const MaterialVariableCellReal& density,
                        const MaterialVariableCellReal& internal_energy,
                        MaterialVariableCellReal& pressure,
                        MaterialVariableCellReal& sound_speed
                        ) = 0;
};

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
