using System;
using Arcane;
using Arcane.Materials;
using Real = System.Double;

[Arcane.Service("DotNetMaterialTest1",typeof(MaterialEos.IMaterialEquationOfState))]
public class StiffedGasMaterialEos : MaterialEos.IMaterialEquationOfState_WrapperService //ArcaneMeshMaterialCSharpUnitTestObject
{
  public StiffedGasMaterialEos(ServiceBuildInfo bi) : base(bi)
  {
  }

  public override void InitEOS(IMeshMaterial mat,
                               MaterialVariableCellReal mat_pressure,
                               MaterialVariableCellReal mat_density,
                               MaterialVariableCellReal mat_internal_energy,
                               MaterialVariableCellReal mat_sound_speed)
  {
    Console.WriteLine("[C#] StiffenedGas Init mat={0} nb_cell={1}",mat.Name(),mat.Cells().Size());
    Real limit_tension = 0.1;
    foreach(MatItem mc in mat){
      Real pressure = mat_pressure[mc];
      Real adiabatic_cst = 1.4;
      Real density = mat_density[mc];
      mat_internal_energy[mc] = (pressure + (adiabatic_cst * limit_tension)) / ((adiabatic_cst - 1.0) * density);
      mat_sound_speed[mc] = System.Math.Sqrt((adiabatic_cst/density)*(pressure+limit_tension));
    }
  }

  public override void ApplyEOS(IMeshMaterial mat,
                                MaterialVariableCellReal mat_density,
                                MaterialVariableCellReal mat_internal_energy,
                                MaterialVariableCellReal mat_pressure,
                                MaterialVariableCellReal mat_sound_speed)
  {
    Console.WriteLine("[C#] StiffenedGas Apply mat={0} nb_cell={1}",mat.Name(),mat.Cells().Size());
    // On met zéro car comme les valeurs calculées sont fictives il peut arriver que
    // la pression soit négative si la tension limite est non nulle.
    Real limit_tension = 0.0;
    foreach(MatItem mc in mat){
      Real internal_energy = mat_internal_energy[mc];
      Real density = mat_density[mc];
      Real adiabatic_cst = 1.4;
      Real pressure = ((adiabatic_cst - 1.0) * density * internal_energy) - (adiabatic_cst * limit_tension);
      mat_pressure[mc] = pressure;
      mat_sound_speed[mc] = System.Math.Sqrt((adiabatic_cst/density)*(pressure+limit_tension));  }
  }
}
