// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "IEquationOfState.h"
#include "PerfectGasEOS_axl.h"

using namespace Arcane;

/**
 * Représente le modèle d'équation d'état <em>Gaz Parfait</em>
 */
class PerfectGasEOSService 
: public ArcanePerfectGasEOSObject
{
 public:
  /** Constructeur de la classe */
  explicit PerfectGasEOSService(const ServiceBuildInfo & sbi)
  : ArcanePerfectGasEOSObject(sbi) {}
  
 public:
  /*! 
   *  Initialise l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et l'énergie interne. 
   */
  void initEOS(const CellGroup& group,
               const VariableCellReal& pressure,
               const VariableCellReal& adiabatic_cst,
               const VariableCellReal& density,
               VariableCellReal& internal_energy,
               VariableCellReal& sound_speed
               ) override;

  /*! 
   *  Applique l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et la pression. 
   */
  void applyEOS(const CellGroup & group,
                const VariableCellReal& adiabatic_cst,
                const VariableCellReal& density,
                const VariableCellReal& internal_energy,
                VariableCellReal& pressure,
                VariableCellReal& sound_speed
                ) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PerfectGasEOSService::
initEOS(const CellGroup& group,
        const VariableCellReal& in_pressure,
        const VariableCellReal& in_adiabatic_cst,
        const VariableCellReal& in_density,
        VariableCellReal& internal_energy,
        VariableCellReal& sound_speed
        )
{
  // Initialise l'énergie et la vitesse du son
  ENUMERATE_CELL(icell, group){
    Real pressure = in_pressure[icell];
    Real adiabatic_cst = in_adiabatic_cst[icell];
    Real density = in_density[icell];
    internal_energy[icell] = pressure / ((adiabatic_cst - 1.0) * density);
    sound_speed[icell] = sqrt(adiabatic_cst * pressure / density);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PerfectGasEOSService::
applyEOS(const CellGroup & group,
         const VariableCellReal& in_adiabatic_cst,
         const VariableCellReal& in_density,
         const VariableCellReal& in_internal_energy,
         VariableCellReal& pressure,
         VariableCellReal& sound_speed
         )
{
  // Calcul de la pression et de la vitesse du son
  ENUMERATE_CELL(icell, group){
    Real internal_energy = in_internal_energy[icell];
    Real density = in_density[icell];
    Real adiabatic_cst = in_adiabatic_cst[icell];
    Real p = (adiabatic_cst - 1.0) * density * internal_energy;
    pressure[icell] = p;
    sound_speed[icell] = sqrt(adiabatic_cst * p / density);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PERFECTGASEOS(PerfectGas, PerfectGasEOSService);
