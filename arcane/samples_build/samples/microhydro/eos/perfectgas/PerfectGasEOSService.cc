// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "PerfectGasEOSService.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PerfectGasEOSService::initEOS(const CellGroup & group)
{
  // Initialise l'énergie et la vitesse du son
  ENUMERATE_CELL(icell, group)
  {
    Real pressure = m_pressure[icell];
    Real adiabatic_cst = m_adiabatic_cst[icell];
    Real density = m_density[icell];
    m_internal_energy[icell] = pressure / ((adiabatic_cst - 1.) * density);
    m_sound_speed[icell] = sqrt(adiabatic_cst * pressure / density);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PerfectGasEOSService::applyEOS(const CellGroup & group)
{
  // Calcul de la pression et de la vitesse du son
  ENUMERATE_CELL(icell, group)
  {
    Real internal_energy = m_internal_energy[icell];
    Real density = m_density[icell];
    Real adiabatic_cst = m_adiabatic_cst[icell];
    Real pressure = (adiabatic_cst - 1.) * density * internal_energy;
    m_pressure[icell] = pressure;
    m_sound_speed[icell] = sqrt(adiabatic_cst * pressure / density);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PERFECTGASEOS(PerfectGas, PerfectGasEOSService);
