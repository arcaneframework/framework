using System;
using Arcane;
using Real = System.Double;
#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Math = Arcane.Math;

[Arcane.Service("PerfectGas",typeof(IEquationOfState))]
public class PerfectGasEOSService
: ArcanePerfectGasEOSObject, IEquationOfState
{
  public PerfectGasEOSService(ServiceBuildInfo sbi)
  : base(sbi)
  {
  }

  public virtual void initEOS(CellGroup group)
  {
    // Initialise l'énergie et la vitesse du son
    foreach(Cell icell in group){
      Real pressure = m_pressure[icell];
      Real adiabatic_cst = m_adiabatic_cst[icell];
      Real density = m_density[icell];
      m_internal_energy[icell] = pressure / ((adiabatic_cst - 1.0) * density);
      m_sound_speed[icell] = Math.Sqrt(adiabatic_cst * pressure / density);
    }
  }

  public virtual void applyEOS(CellGroup group)
  {
    //Console.WriteLine("GLOBAL TIME={0}",m_global_time.Value);
    // Calcul de la pression et de la vitesse du son
    foreach(Cell icell in group){
      Real internal_energy = m_internal_energy[icell];
      Real density = m_density[icell];
      Real adiabatic_cst = m_adiabatic_cst[icell];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      m_pressure[icell] = pressure;
      m_sound_speed[icell] = Math.Sqrt(adiabatic_cst * pressure / density);
    }
  }

  public virtual void applyEOS2(CellGroup group)
  {
    ItemIndexArrayView local_ids = group.View().Indexes;
    int nb_cell = local_ids.Length;
    for( int i=0; i<nb_cell; ++i ){
      Int32 icell = local_ids[i];
      Real internal_energy = m_internal_energy[icell];
      Real density = m_density[icell];
      Real adiabatic_cst = m_adiabatic_cst[icell];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      m_pressure[icell] = pressure;
      m_sound_speed[icell] = Math.Sqrt(adiabatic_cst * pressure / density);
    }
  }
}
