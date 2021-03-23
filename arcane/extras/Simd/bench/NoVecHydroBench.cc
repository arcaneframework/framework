#include "bench/Wrapper.h"

#include "arcane/utils/ArrayView.h"

#include "bench/Variables.h"
#include "bench/Mesh.h"
#include "bench/HydroBenchBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __INTEL_COMPILER
#define PRAGMA_IVDEP _Pragma("ivdep")
#else
#ifdef __GNUC__
#define PRAGMA_IVDEP _Pragma("GCC ivdep")
#endif
#endif

//#undef PRAGMA_IVDEP

#ifndef PRAGMA_IVDEP
#define PRAGMA_IVDEP
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NoVecHydroBench
: public HydroBenchBase
{
 public:

  void _computeCQs(Real3 node_coord[8],Real3 face_coord[6],Cell cell);

  virtual void computeGeometric(Int32 nb_compute_cqs);
  virtual void computeEquationOfState(Int32 nb_compute);
  virtual const char* getSimdName() const { return "NOVEC"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NoVecHydroBench::
_computeCQs(Real3 node_coord[8],Real3 face_coord[6],Cell cell)
{
  const Real3 c0 = face_coord[0];
  const Real3 c1 = face_coord[1];
  const Real3 c2 = face_coord[2];
  const Real3 c3 = face_coord[3];
  const Real3 c4 = face_coord[4];
  const Real3 c5 = face_coord[5];

  const Real demi = ARCANE_REAL(0.5);
  const Real five = ARCANE_REAL(5.0);

  // Calcul des normales face 1 :
  const Real3 n1a04 = demi * math::cross(node_coord[0] - c0 , node_coord[3] - c0);
  const Real3 n1a03 = demi * math::cross(node_coord[3] - c0 , node_coord[2] - c0);
  const Real3 n1a02 = demi * math::cross(node_coord[2] - c0 , node_coord[1] - c0);
  const Real3 n1a01 = demi * math::cross(node_coord[1] - c0 , node_coord[0] - c0);

  // Calcul des normales face 2 :
  const Real3 n2a05 = demi * math::cross(node_coord[0] - c1 , node_coord[4] - c1);
  const Real3 n2a12 = demi * math::cross(node_coord[4] - c1 , node_coord[7] - c1);
  const Real3 n2a08 = demi * math::cross(node_coord[7] - c1 , node_coord[3] - c1);
  const Real3 n2a04 = demi * math::cross(node_coord[3] - c1 , node_coord[0] - c1);

  // Calcul des normales face 3 :
  const Real3 n3a01 = demi * math::cross(node_coord[0] - c2 , node_coord[1] - c2);
  const Real3 n3a06 = demi * math::cross(node_coord[1] - c2 , node_coord[5] - c2);
  const Real3 n3a09 = demi * math::cross(node_coord[5] - c2 , node_coord[4] - c2);
  const Real3 n3a05 = demi * math::cross(node_coord[4] - c2 , node_coord[0] - c2);

  // Calcul des normales face 4 :
  const Real3 n4a09 = demi * math::cross(node_coord[4] - c3 , node_coord[5] - c3);
  const Real3 n4a10 = demi * math::cross(node_coord[5] - c3 , node_coord[6] - c3);
  const Real3 n4a11 = demi * math::cross(node_coord[6] - c3 , node_coord[7] - c3);
  const Real3 n4a12 = demi * math::cross(node_coord[7] - c3 , node_coord[4] - c3);
	
  // Calcul des normales face 5 :
  const Real3 n5a02 = demi * math::cross(node_coord[1] - c4 , node_coord[2] - c4);
  const Real3 n5a07 = demi * math::cross(node_coord[2] - c4 , node_coord[6] - c4);
  const Real3 n5a10 = demi * math::cross(node_coord[6] - c4 , node_coord[5] - c4);
  const Real3 n5a06 = demi * math::cross(node_coord[5] - c4 , node_coord[1] - c4);
      
  // Calcul des normales face 6 :
  const Real3 n6a03 = demi * math::cross(node_coord[2] - c5 , node_coord[3] - c5);
  const Real3 n6a08 = demi * math::cross(node_coord[3] - c5 , node_coord[7] - c5);
  const Real3 n6a11 = demi * math::cross(node_coord[7] - c5 , node_coord[6] - c5);
  const Real3 n6a07 = demi * math::cross(node_coord[6] - c5 , node_coord[2] - c5);

  const Real real_1div12 = 1.0 / 12.0;

  // Calcul des résultantes aux sommets :
  m_cell_cqs[cell][0] = (five*(n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
                         (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09))*real_1div12;
  m_cell_cqs[cell][1] = (five*(n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
                         (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07))*real_1div12;
  m_cell_cqs[cell][2] = (five*(n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
                         (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08))*real_1div12;
  m_cell_cqs[cell][3] = (five*(n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
                         (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11))*real_1div12;
  m_cell_cqs[cell][4] = (five*(n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
                         (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11))*real_1div12;
  m_cell_cqs[cell][5] = (five*(n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
                         (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02))*real_1div12;
  m_cell_cqs[cell][6] = (five*(n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
                         (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08))*real_1div12;
  m_cell_cqs[cell][7] = (five*(n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
                         (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03))*real_1div12;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NoVecHydroBench::
computeGeometric(Int32 nb_compute_cqs)
{
  // Copie locale des coordonnées des sommets d'une maille
  Real3 coord[8];
  // Coordonnées des centres des faces
  Real3 face_coord[6];

  ENUMERATE_CELL(icell,m_mesh.m_cells_vector){
    Cell cell = *icell;

    // Recopie les coordonnées locales (pour le cache)
    for( NodeEnumerator i_node(cell.nodes()); i_node.index()<8; ++i_node )
      coord[i_node.index()] = m_node_coord[i_node];

    // Calcul les coordonnées des centres des faces
    face_coord[0] = 0.25 * ( coord[0] + coord[3] + coord[2] + coord[1] );
    face_coord[1] = 0.25 * ( coord[0] + coord[4] + coord[7] + coord[3] );
    face_coord[2] = 0.25 * ( coord[0] + coord[1] + coord[5] + coord[4] );
    face_coord[3] = 0.25 * ( coord[4] + coord[5] + coord[6] + coord[7] );
    face_coord[4] = 0.25 * ( coord[1] + coord[2] + coord[6] + coord[5] );
    face_coord[5] = 0.25 * ( coord[2] + coord[3] + coord[7] + coord[6] );

    // Calcule les résultantes aux sommets
    for( int xz=0; xz<nb_compute_cqs; ++xz)
      _computeCQs(coord,face_coord,cell);

  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NoVecHydroBench::
computeEquationOfState(Int32 nb_compute)
{
  for( Int32 iloop=0; iloop<nb_compute; ++iloop){

    PRAGMA_IVDEP
    ENUMERATE_CELL(icell,m_mesh.m_cells_vector){
      Real adiabatic_cst = m_adiabatic_cst[icell];
      Real volume_ratio = m_volume[icell] / m_old_volume[icell];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = m_internal_energy[icell];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      m_internal_energy[icell] = internal_energy;
      Real density = m_density[icell];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy; 
      Real sound_speed = math::sqrt(adiabatic_cst*pressure/density);
      m_pressure[icell] = pressure;
      m_sound_speed[icell] = sound_speed;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Arcane::HydroBenchBase*
createNoVecHydroBench()
{
  return new Arcane::NoVecHydroBench();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
