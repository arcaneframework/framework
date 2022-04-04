// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef IEQUATIONOFSTATE_H
#define IEQUATIONOFSTATE_H

#include <arcane/ItemTypes.h>
#include <arcane/VariableTypedef.h>

using namespace Arcane;

namespace EOS
{
//! Interface du service du modèle de calcul de l'équation d'état.
class IEquationOfState
{
 public:
  /** Destructeur de la classe */
  virtual ~IEquationOfState() = default;
  
 public:
  /*!
   *  Initialise l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et l'énergie interne. 
   */
  virtual void initEOS(const CellGroup& group,
                       const VariableCellReal& pressure,
                       const VariableCellReal& adiabatic_cst,
                       const VariableCellReal& density,
                       VariableCellReal& internal_energy,
                       VariableCellReal& sound_speed
                       ) =0;
  /*!
   *  Applique l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et la pression. 
   */
  virtual void applyEOS(const CellGroup & group,
                        const VariableCellReal& adiabatic_cst,
                        const VariableCellReal& density,
                        const VariableCellReal& internal_energy,
                        VariableCellReal& pressure,
                        VariableCellReal& sound_speed
                        ) = 0;
};

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
