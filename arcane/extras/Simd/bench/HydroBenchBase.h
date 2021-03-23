#ifndef BENCH_SIMD_HYDROBENCHBASE_H
#define BENCH_SIMD_HYDROBENCHBASE_H

#include <arcane/utils/ArcaneGlobal.h>

#include "bench/Wrapper.h"
#include "arcane/utils/ArrayView.h"

#include "bench/Variables.h"
#include "bench/Mesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
inline Real
dot(Real3 u,Real3 v)
{
  return (u.x * v.x  +  u.y * v.y  +  u.z * v.z);
}
inline Real3
cross(Real3 v1, Real3 v2)
{
  Real3 v;
  v.x = v1.y*v2.z - v1.z*v2.y;
  v.y = v2.x*v1.z - v2.z*v1.x;
  v.z = v1.x*v2.y - v1.y*v2.x;
  
  return v;
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HydroBenchBase
{
 public:
  
  Int32 nbCell() { return m_mesh.m_cells_vector.size(); }

  void allocate(Int32 nb_x,Int32 nb_y,Int32 nb_z)
  {
    m_mesh.generateMesh(nb_x,nb_y,nb_z,m_node_coord);
    Int32 nb_cell = m_mesh.m_nb_cell;

    m_cell_cqs.resize(nb_cell,8);

    m_adiabatic_cst.resize(nb_cell);
    m_volume.resize(nb_cell);
    m_density.resize(nb_cell);
    m_old_volume.resize(nb_cell);
    m_internal_energy.resize(nb_cell);
    m_cell_mass.resize(nb_cell);
    m_sound_speed.resize(nb_cell);
    m_pressure.resize(nb_cell);
  }

  // Must be called after computeGeometric()
  void initForEquationOfState()
  {
    Real ref_volume = m_mesh.m_reference_volume;

    ENUMERATE_CELL(icell,m_mesh.m_cells_vector){
      Cell cell(*icell);
      Real volume = 0.0;
      for( NodeEnumerator inode(cell.nodes()); inode.index()<8; ++inode ){
        //coord[i_node.index()].set(si,m_node_coord[i_node]);
        volume += math::dot(m_node_coord[inode],m_cell_cqs[icell][inode.index()]);
      }
      volume /= 3.0;
          
      m_volume[cell] = volume;
      m_old_volume[cell] = ref_volume;
      
      m_adiabatic_cst[icell] = 1.4;
      m_density[icell] = 2.0;
      m_pressure[icell] = 1.0;

      m_internal_energy[icell] = m_pressure[icell] / ((m_adiabatic_cst[icell]-1.0) * m_density[icell]);
    }
  }

 public:

  virtual const char* getSimdName() const =0;
  virtual void computeGeometric(Int32 nb_compute_cqs) =0;
  virtual void computeEquationOfState(Int32 nb_compute) =0;

  void compare(HydroBenchBase* rhs)
  {
    _compareVariable(m_volume,rhs->m_volume,"Volume");
    _compareVariable(m_old_volume,rhs->m_old_volume,"OldVolume");
    _compareVariable(m_density,rhs->m_density,"Density");
    _compareVariable(m_pressure,rhs->m_pressure,"Pressure");
    _compareVariable(m_internal_energy,rhs->m_internal_energy,"InternalEnergy");
    _compareVariable(m_sound_speed,rhs->m_sound_speed,"SoundSpeed");
  }

  void _compareVariable(VariableCellScalarReal& var1,VariableCellScalarReal& var2,const char* name)
  {
    ConstArrayView<Real> var1v = var1.asArray();
    ConstArrayView<Real> var2v = var2.asArray();
    Int32 n = var1v.size();
    int nb_diff = 0;
    for( Int32 i=0; i<n; ++i ){
      Real diff = math::abs(var1v[i]-var2v[i]);
      if (diff>1e-15){
        if (nb_diff<10)
          printf("DIFF i=%d ref=%lf v=%lf diff=%e\n",i,var1v[i],var2v[i],(var2v[i]-var1v[i]));

        ++nb_diff;
      }
    }
    if (nb_diff!=0)
      printf("WARNING! VAR %s NB_DIFF=%d\n",name,nb_diff);
  }

 protected:
    
  Mesh m_mesh;
  VariableNodeScalarReal3 m_node_coord;
  VariableCellArrayReal3 m_cell_cqs;

  VariableCellScalarReal m_adiabatic_cst;
  VariableCellScalarReal m_volume;
  VariableCellScalarReal m_density;
  VariableCellScalarReal m_old_volume;
  VariableCellScalarReal m_internal_energy;
  VariableCellScalarReal m_cell_mass;
  VariableCellScalarReal m_sound_speed;
  VariableCellScalarReal m_pressure;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
