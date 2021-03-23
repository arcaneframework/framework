#include "bench/Mesh.h"
#include <stdlib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Mesh::
Mesh()
: m_seed(13)
{
  m_ydelta = 0.025;
  m_zdelta = 0.02;
  m_xdelta = 0.022;
  m_reference_volume = m_xdelta*m_ydelta*m_zdelta;
}

inline Real Mesh::
_getRand()
{
  int v = rand_r(&m_seed);
  double x = (double)v / (double)RAND_MAX;
  return x * 0.002;
}

void Mesh::
generateMesh(Int64 nb_cell_x,Int64 nb_cell_y,Int64 nb_cell_z,VariableNodeScalarReal3& nodes_coord)
{ 
  typedef ItemInternal* ItemInternalPtr;

  Int32 nb_node_x  = nb_cell_x + 1;
  Int32 nb_node_y =  nb_cell_y + 1;
  Int32 nb_node_z  = nb_cell_z + 1;
  Int32 nb_node_xy = nb_node_x * nb_node_y;
  Int32 mesh_nb_cell = nb_cell_x * nb_cell_y * nb_cell_z;
  Int32 mesh_nb_node = nb_node_x * nb_node_y * nb_node_z;

  m_nb_cell = mesh_nb_cell;
  m_nb_node = mesh_nb_node;

  m_cells.resize(mesh_nb_cell);
  m_cells_internal.resize(mesh_nb_cell);
  for( Integer i=0; i<mesh_nb_cell; ++i )
    m_cells_internal[i] = &m_cells[i];

  m_nodes.resize(mesh_nb_node);
  m_nodes_internal.resize(mesh_nb_node);
  for( Integer i=0; i<mesh_nb_node; ++i )
    m_nodes_internal[i] = &m_nodes[i];
  ItemInternalPtr* nodes_begin = &m_nodes_internal[0];

  nodes_coord.resize(mesh_nb_node);
  ArrayView<Real3> nodes_coord_array = nodes_coord.asArray();

  Real ydelta = m_ydelta;
  Real zdelta = m_zdelta;
  Real xdelta = m_xdelta;
    
  for(Int32 z=0; z<nb_node_z; ++z ){
    for(Int32 y=0; y<nb_node_y; ++y ){
      for(Int32 x=0; x<nb_node_x; ++x ){
        Real nx = (1 + _getRand()) * xdelta * (Real)(x);
        Real ny = (1 + _getRand()) * ydelta * (Real)(y);
        Real nz = (1 + _getRand()) * zdelta * (Real)(z);
        Int32 node_unique_id = x + y*nb_node_x + (z)*nb_node_x*nb_node_y;
          
        m_nodes[node_unique_id].m_local_id = node_unique_id;
        nodes_coord_array[node_unique_id] = Real3(nx,ny,nz);
      }
    }
  }
  
  m_cells_connectivity.resize(mesh_nb_cell*8);

  // Create cells
  for( Integer z=0; z<nb_cell_z; ++z ){
    for( Integer y=0; y<nb_cell_y; ++y ){
      for( Integer x=0; x<nb_cell_x; ++x ){

        Int32 cell_unique_id = x + y*nb_cell_x + (z)*nb_cell_y*nb_cell_x;

        ItemInternal& cell_internal = m_cells[cell_unique_id];
        cell_internal.m_local_id = cell_unique_id;
        cell_internal.m_nb_sub_item = 8;
        cell_internal.m_sub_internals = nodes_begin;
          
        Integer base_id = x + y*nb_node_x + z*nb_node_xy;

        Int32* sub_items = &m_cells_connectivity[cell_unique_id*8];
        cell_internal.m_sub_items_lid = sub_items;

        sub_items[0] = base_id;
        sub_items[1] = base_id                          + 1;
        sub_items[2] = base_id              + nb_node_x + 1;
        sub_items[3] = base_id              + nb_node_x + 0;
        sub_items[4] = base_id + nb_node_xy;
        sub_items[5] = base_id + nb_node_xy             + 1;
        sub_items[6] = base_id + nb_node_xy + nb_node_x + 1;
        sub_items[7] = base_id + nb_node_xy + nb_node_x + 0;
      }
    }
  }

  Array<Int32> nodes_lid(mesh_nb_node);
  for( Int32 i=0; i<mesh_nb_node; ++i )
    nodes_lid[i] = i;
  m_nodes_vector = NodeVector(m_nodes_internal.view(),nodes_lid.view());
      
  Array<Int32> cells_lid(mesh_nb_cell);
  for( Int32 i=0; i<mesh_nb_cell; ++i )
    cells_lid[i] = i;
  m_cells_vector = CellVector(m_cells_internal.view(),cells_lid.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
