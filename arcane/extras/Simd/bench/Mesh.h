#ifndef BENCH_SIMD_MESH_H
#define BENCH_SIMD_MESH_H

#include "Wrapper.h"
#include "Variables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Mesh
{
  
  Array<ItemInternal> m_nodes;
  Array<ItemInternal> m_cells;
  Array<Int32> m_cells_connectivity;
  Array<ItemInternal*> m_nodes_internal;
  Array<ItemInternal*> m_cells_internal;

  unsigned int m_seed;
  Real m_ydelta;
  Real m_zdelta;
  Real m_xdelta;

 public:

  Int32 m_nb_cell;
  Int32 m_nb_node;
  ItemVectorT<Node> m_nodes_vector;
  ItemVectorT<Cell> m_cells_vector;
  Real m_reference_volume;

 public:

  Mesh();
  void generateMesh(Int64 nb_cell_x,Int64 nb_cell_y,Int64 nb_cell_z,VariableNodeScalarReal3& nodes_coord);
  inline Real _getRand();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
