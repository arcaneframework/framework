//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;
using Real = System.Double;

// TRES IMPORTANT: la classe doit être publique
[Arcane.Service("PerfectGasCS",typeof(EOS.IEquationOfState))]
public class PerfectGasCS : EOS.IEquationOfState_WrapperService
{
  static PerfectGasCS()
  {
    Console.WriteLine("STATIC INIT");
  }

  public PerfectGasCS(ServiceBuildInfo bi) : base(bi)
  {
  }

  public override void InitEOS(CellGroup group,
                               VariableCellReal in_pressure,
                               VariableCellReal in_adiabatic_cst,
                               VariableCellReal in_density,
                               VariableCellReal internal_energy,
                               VariableCellReal sound_speed)
  {
    Console.WriteLine("[C#] Initialize EOS 'PerfectGasCS'");
    // Initialise l'énergie et la vitesse du son
    foreach(Cell icell in group){
      Real pressure = in_pressure[icell];
      Real adiabatic_cst = in_adiabatic_cst[icell];
      Real density = in_density[icell];
      internal_energy[icell] = pressure / ((adiabatic_cst - 1.0) * density);
      sound_speed[icell] = System.Math.Sqrt(adiabatic_cst * pressure / density);
    }
  }

  public override void ApplyEOS(CellGroup group,
                                VariableCellReal in_adiabatic_cst,
                                VariableCellReal in_density,
                                VariableCellReal in_internal_energy,
                                VariableCellReal pressure,
                                VariableCellReal sound_speed)
  {
    Console.WriteLine("[C#] Apply EOS 'PerfectGasCS'");
    foreach(Cell icell in group){
      Real internal_energy = in_internal_energy[icell];
      Real density = in_density[icell];
      Real adiabatic_cst = in_adiabatic_cst[icell];
      Real p = (adiabatic_cst - 1.0) * density * internal_energy;
      pressure[icell] = p;
      sound_speed[icell] = System.Math.Sqrt(adiabatic_cst * p / density);
    }
  }
}
