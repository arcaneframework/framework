#include <arcane/utils/ArcaneGlobal.h>

#include "bench/Wrapper.h"
#include "arcane/utils/ArrayView.h"

ARCANE_BEGIN_NAMESPACE
class BadAlignmentException
: public std::exception
{
 public:
  virtual const char* what() const ARCANE_NOEXCEPT { return "Bad alignment"; }
};
ARCANE_END_NAMESPACE

#define ARCANE_SIMD_BENCH
#define A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) ::Arcane::_EnumeratorClassName
#define A_TRACE_ENUMERATOR_WHERE

#include <arcane/SimdItem.h>
#include <arcane/SimdMathUtils.h>

#include "bench/Variables.h"
#include "bench/Mesh.h"
#include "bench/HydroBenchBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
class ItemVariableView
{
 public:
  ItemVariableView(ARCANE_RESTRICT DataType* v) : values(v){}
  SimdSetter<DataType> operator[](SimdItemIndexT<ItemType> simd_item) const
  {
    return SimdSetter<DataType>(values,simd_item.simdLocalIds());
  }
 private:
  ARCANE_RESTRICT DataType* values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
class ItemVariableConstView
{
 public:
  ItemVariableConstView(ConstArrayView<DataType> v) : values(v){}

  typename SimdTypeTraits<DataType>::SimdType operator[](SimdItemIndexT<ItemType> simd_item) const
  {
    typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
    return SimdType(values.begin(),simd_item.simdLocalIds());
  }

 private:
  ConstArrayView<DataType> values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
ItemVariableView<ItemType,DataType>
getview(MeshVariableScalarRefT<ItemType,DataType>& var)
{
  return ItemVariableView<ItemType,DataType>(var.asArray().unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
ItemVariableConstView<ItemType,DataType>
getconstview(const MeshVariableScalarRefT<ItemType,DataType>& var)
{
  return ItemVariableConstView<ItemType,DataType>(var.asArray());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimdHydroBench
: public HydroBenchBase
{
 public:
  void _computeCQsSimd(SimdReal3 node_coord[8],SimdReal3 face_coord[6],SimdReal3 cqs[8]);
  virtual void computeGeometric(Int32 nb_compute_cqs);
  virtual void computeEquationOfState(Int32 nb_compute);
  virtual const char* getSimdName() const { return SimdInfo::name(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimdHydroBench::
_computeCQsSimd(SimdReal3 node_coord[8],SimdReal3 face_coord[6],SimdReal3 cqs[8])
{
  const SimdReal3 c0 = face_coord[0];
  const SimdReal3 c1 = face_coord[1];
  const SimdReal3 c2 = face_coord[2];
  const SimdReal3 c3 = face_coord[3];
  const SimdReal3 c4 = face_coord[4];
  const SimdReal3 c5 = face_coord[5];

  const Real demi = ARCANE_REAL(0.5);
  const Real five = ARCANE_REAL(5.0);

  // Calcul des normales face 1 :
  const SimdReal3 n1a04 = demi * math::cross(node_coord[0] - c0 , node_coord[3] - c0);
  const SimdReal3 n1a03 = demi * math::cross(node_coord[3] - c0 , node_coord[2] - c0);
  const SimdReal3 n1a02 = demi * math::cross(node_coord[2] - c0 , node_coord[1] - c0);
  const SimdReal3 n1a01 = demi * math::cross(node_coord[1] - c0 , node_coord[0] - c0);

  // Calcul des normales face 2 :
  const SimdReal3 n2a05 = demi * math::cross(node_coord[0] - c1 , node_coord[4] - c1);
  const SimdReal3 n2a12 = demi * math::cross(node_coord[4] - c1 , node_coord[7] - c1);
  const SimdReal3 n2a08 = demi * math::cross(node_coord[7] - c1 , node_coord[3] - c1);
  const SimdReal3 n2a04 = demi * math::cross(node_coord[3] - c1 , node_coord[0] - c1);

  // Calcul des normales face 3 :
  const SimdReal3 n3a01 = demi * math::cross(node_coord[0] - c2 , node_coord[1] - c2);
  const SimdReal3 n3a06 = demi * math::cross(node_coord[1] - c2 , node_coord[5] - c2);
  const SimdReal3 n3a09 = demi * math::cross(node_coord[5] - c2 , node_coord[4] - c2);
  const SimdReal3 n3a05 = demi * math::cross(node_coord[4] - c2 , node_coord[0] - c2);

  // Calcul des normales face 4 :
  const SimdReal3 n4a09 = demi * math::cross(node_coord[4] - c3 , node_coord[5] - c3);
  const SimdReal3 n4a10 = demi * math::cross(node_coord[5] - c3 , node_coord[6] - c3);
  const SimdReal3 n4a11 = demi * math::cross(node_coord[6] - c3 , node_coord[7] - c3);
  const SimdReal3 n4a12 = demi * math::cross(node_coord[7] - c3 , node_coord[4] - c3);
	
  // Calcul des normales face 5 :
  const SimdReal3 n5a02 = demi * math::cross(node_coord[1] - c4 , node_coord[2] - c4);
  const SimdReal3 n5a07 = demi * math::cross(node_coord[2] - c4 , node_coord[6] - c4);
  const SimdReal3 n5a10 = demi * math::cross(node_coord[6] - c4 , node_coord[5] - c4);
  const SimdReal3 n5a06 = demi * math::cross(node_coord[5] - c4 , node_coord[1] - c4);
      
  // Calcul des normales face 6 :
  const SimdReal3 n6a03 = demi * math::cross(node_coord[2] - c5 , node_coord[3] - c5);
  const SimdReal3 n6a08 = demi * math::cross(node_coord[3] - c5 , node_coord[7] - c5);
  const SimdReal3 n6a11 = demi * math::cross(node_coord[7] - c5 , node_coord[6] - c5);
  const SimdReal3 n6a07 = demi * math::cross(node_coord[6] - c5 , node_coord[2] - c5);

  const Real real_1div12 = 1.0 / 12.0;

  cqs[0] = (five*(n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
            (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09))*real_1div12;
  cqs[1] = (five*(n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
            (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07))*real_1div12;
  cqs[2] = (five*(n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
            (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08))*real_1div12;
  cqs[3] = (five*(n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
            (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11))*real_1div12;
  cqs[4] = (five*(n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
            (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11))*real_1div12;
  cqs[5] = (five*(n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
            (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02))*real_1div12;
  cqs[6] = (five*(n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
            (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08))*real_1div12;
  cqs[7] = (five*(n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
            (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03))*real_1div12;
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimdHydroBench::
computeGeometric(Int32 nb_compute_cqs)
{
  // Copie locale des coordonnées des sommets d'une maille
  SimdReal3 coord[8];
  // Coordonnées des centres des faces
  SimdReal3 face_coord[6];

  SimdReal3 cqs[8];

  ENUMERATE_SIMD_CELL(ivecitem,m_mesh.m_cells_vector){
    SimdCell vitem = *ivecitem;

    for( Integer si=0; si<SimdReal::BLOCK_SIZE; ++si ){
      Cell cell(vitem[si]);
      // Copy node coordinates in local ' coord'.
      for( NodeEnumerator i_node(cell.nodes()); i_node.index()<8; ++i_node ){
        coord[i_node.index()].set(si,m_node_coord[i_node]);
      }
    }

    // Compute center of faces.
    face_coord[0] = ARCANE_REAL(0.25) * ( coord[0] + coord[3] + coord[2] + coord[1] );
    face_coord[1] = ARCANE_REAL(0.25) * ( coord[0] + coord[4] + coord[7] + coord[3] );
    face_coord[2] = ARCANE_REAL(0.25) * ( coord[0] + coord[1] + coord[5] + coord[4] );
    face_coord[3] = ARCANE_REAL(0.25) * ( coord[4] + coord[5] + coord[6] + coord[7] );
    face_coord[4] = ARCANE_REAL(0.25) * ( coord[1] + coord[2] + coord[6] + coord[5] );
    face_coord[5] = ARCANE_REAL(0.25) * ( coord[2] + coord[3] + coord[7] + coord[6] );
      
    for( int xz=0; xz<nb_compute_cqs; ++xz)
      _computeCQsSimd(coord,face_coord,cqs);

    // Copie back 'cqs' to corresponding variable.
    ENUMERATE_SIMD_ITEM(si,ivecitem){
      Cell cell(vitem[si]);
      ArrayView<Real3> cqsv = m_cell_cqs[cell];
      for( Integer i_node=0; i_node<8; ++i_node )
        cqsv[i_node] = cqs[i_node][si];
    }
      
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimdHydroBench::
computeEquationOfState(Int32 nb_compute)
{
  auto in_adiabatic_cst = getconstview(m_adiabatic_cst);
  auto in_volume = getconstview(m_volume);
  auto in_density = getconstview(m_density);
  auto in_old_volume = getconstview(m_old_volume);
  auto in_internal_energy = getconstview(m_internal_energy);

  auto out_internal_energy = getview(m_internal_energy);
  auto out_sound_speed = getview(m_sound_speed);
  auto out_pressure = getview(m_pressure);

  for( Int32 iloop=0; iloop<nb_compute; ++iloop){

    ENUMERATE_SIMD_CELL(icell,m_mesh.m_cells_vector){
      SimdReal adiabatic_cst = in_adiabatic_cst[icell];
      SimdReal volume_ratio = in_volume[icell] / in_old_volume[icell];
      SimdReal x = 0.5 * adiabatic_cst - 1.0;
      SimdReal numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      SimdReal denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      SimdReal internal_energy = in_internal_energy[icell];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[icell] = internal_energy;
      SimdReal density = in_density[icell];
      SimdReal pressure = (adiabatic_cst - 1.0) * density * internal_energy; 
      SimdReal sound_speed = math::sqrt(adiabatic_cst*pressure/density);
      out_pressure[icell] = pressure;
      out_sound_speed[icell] = sound_speed;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Arcane::HydroBenchBase*
createHydroBench()
{
  return new Arcane::SimdHydroBench();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
