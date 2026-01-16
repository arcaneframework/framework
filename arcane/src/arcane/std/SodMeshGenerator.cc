// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SodMeshGenerator.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Service de génération d'un maillage à-la 'sod'.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IMeshReader.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/Service.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IMeshBuilder.h"

#include "arcane/std/IMeshGenerator.h"
#include "arcane/std/SodMeshGenerator.h"
#include "arcane/std/internal/SodStandardGroupsBuilder.h"

#include "arcane/std/Sod3DMeshGenerator_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SodMeshGenerator::Impl
: public TraceAccessor
{
 public:
  Impl(ITraceMng* tm,bool zyx,Int32 wanted_x,Int32 wanted_y,Int32 wanted_z,
       bool z_is_total,Int32 mesh_dimension,Real random_coef,
       Real delta0,Real delta1,Real delta2)
  : TraceAccessor(tm),
    m_wanted_x(wanted_x), m_wanted_y(wanted_y), m_wanted_z(wanted_z),
    m_zyx_generate(zyx), m_z_is_total(z_is_total), m_mesh_dimension(mesh_dimension),
    m_random_coef(random_coef)
  {
    m_xyz_delta[0] = delta0;
    m_xyz_delta[1] = delta1;
    m_xyz_delta[2] = delta2;
  }

 public:
  IntegerConstArrayView communicatingSubDomains() const
  {
    return m_communicating_sub_domains;
  }
  bool generateMesh(IPrimaryMesh* mesh);

 private:

  Integer m_wanted_x;
  Integer m_wanted_y;
  Integer m_wanted_z;
  UniqueArray<Integer> m_communicating_sub_domains;
  bool m_zyx_generate; //!< \a true si on génère en z, puis y et enfin x
  bool m_z_is_total; //!< \a true si z est le nombre total de couche pour l'ensemble des procs.
  Real m_xyz_delta[3]; //!< \a les deltas
  int m_mesh_dimension;
  Real m_random_coef;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SodMeshGenerator::
SodMeshGenerator(IPrimaryMesh* mesh,bool zyx)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_zyx_generate(zyx)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SodMeshGenerator::
~SodMeshGenerator()
{
  // Nécessaire pour std::unique_ptr.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IntegerConstArrayView SodMeshGenerator::
communicatingSubDomains() const
{
  if (m_p.get())
    return m_p->communicatingSubDomains();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SodMeshGenerator::
readOptions(XmlNode node)
{
  Int32 wanted_x = node.child("x").valueAsInteger();
  Int32 wanted_y = 0;
  Int32 wanted_z = 0;
  Int32 mesh_dimension = 1;
  XmlNode x_node = node.child("x");
  XmlNode y_node = node.child("y");
  XmlNode z_node = node.child("z");
  if (z_node.null() && y_node.null()){
    info() << "[SodMeshGenerator] 1D";
    mesh_dimension = 1;
    wanted_z = 1;
    wanted_y = 1;
  }
  else if (z_node.null()){
    info() << "[SodMeshGenerator] 2D";
    // WARNING: In 2D Z and Y are switched
    mesh_dimension = 2;
    //z_node = node.child("y");
    wanted_z = y_node.valueAsInteger();
    wanted_y = 1;
  }
  else{
    info() << "[SodMeshGenerator] 3D";
    mesh_dimension = 3;
    wanted_y = y_node.valueAsInteger();
    wanted_z = z_node.valueAsInteger();
  }

  bool z_is_total = z_node.attr("total").valueAsBoolean();
   
  Real delta0 = x_node.attr("delta").valueAsReal();
  Real delta1 = y_node.attr("delta").valueAsReal();
  Real delta2 = z_node.attr("delta").valueAsReal();
  
  Real random_coef = node.child("random-coef").valueAsReal();
  if (wanted_x==0 || wanted_y==0 || wanted_z==0){
    info() << "Bad syntax for <meshgenerator>";
    info() << "Expected is <meshgenerator><sod><x>nx</x><y>ny</y><z>nz</z></sod></meshgenerator>";
    info() << "or <meshgenerator><sod><x>nx</x><y>ny</y></sod></meshgenerator>";
    info() << "or <meshgenerator><sod><x>nx</x></sod></meshgenerator>";
    error() << "Bad syntax for <meshgenerator>";
    return true;
  }
  m_p = std::make_unique<Impl>(traceMng(),m_zyx_generate,wanted_x,wanted_y,wanted_z,z_is_total,
                               mesh_dimension,random_coef,delta0,delta1,delta2);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Génère un UniqueId à partir des indices de bloc (x,y,z).
 *
 * S'assure qu'il n'y aura pas de problèmes de débordement entre les Integer
 * et les Int64.
 */
inline Int64
_generateCellUniqueId(Integer x,Integer y,Integer z,Integer nb_y,
                      Int64 first_z,Int64 total_para_z)
{
  Int64 ax = x;
  Int64 ay = y;
  Int64 az = z;
  Int64 anb_y = nb_y;

  Int64 unique_id = ay + (az+first_z)*anb_y + ax*anb_y*total_para_z;
  return unique_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SodMeshGenerator::
generateMesh()
{
  return m_p->generateMesh(m_mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SodMeshGenerator::Impl::
generateMesh(IPrimaryMesh* mesh)
{
  info() << "Generate Mesh from SodMeshGenerator";

  IParallelMng* pm = mesh->parallelMng();
  bool is_parallel   = pm->isParallel();
  Int32 nb_part = pm->commSize();
  if (is_parallel){
    if (m_z_is_total)
      m_wanted_z /= nb_part;
  }
  Integer nb_cell_x = m_wanted_x;
  Integer nb_cell_y = m_wanted_y;
  Integer nb_cell_z = m_wanted_z;
  info() << "nb x=" << nb_cell_x << " y=" << nb_cell_y << " z=" << nb_cell_z;

  // Positionne des propriétés sur le maillage pour qu'il puisse connaître
  // le nombre de mailles dans chaque direction. Cela est utilisé
  // notammement par CartesianMesh.
  Properties* mesh_properties = mesh->properties();
  mesh_properties->setInt64("GlobalNbCellX",nb_cell_x);
  mesh_properties->setInt64("GlobalNbCellY",nb_cell_y);
  mesh_properties->setInt64("GlobalNbCellZ",nb_cell_z);

  Int32 my_part = pm->commRank();
  bool is_first_proc = (my_part==0);
  bool is_last_proc  = ((1+my_part)==nb_part);
  Int64 total_para_cell_z = ((Int64)nb_cell_z) * ((Int64)nb_part);
  Int64 total_para_node_z = total_para_cell_z+1;
  Int64 first_cell_z = 0;
  Int64 first_node_z = 0;
  if (is_parallel){
    if (is_first_proc || is_last_proc){
      m_communicating_sub_domains.resize(1);
      first_cell_z = nb_cell_z*my_part;
      if (is_first_proc){
        m_communicating_sub_domains[0] = my_part+1;
        first_cell_z = 0;
      }
      else if (is_last_proc){
        m_communicating_sub_domains[0] = my_part-1;
      }
      else{
      }
      first_node_z = first_cell_z;
    }
    else{
      m_communicating_sub_domains.resize(2);
      m_communicating_sub_domains[0] = my_part-1;
      m_communicating_sub_domains[1] = my_part+1;
      first_cell_z = (nb_cell_z*my_part);
      first_node_z = first_cell_z;
    }
  }

  //Integer nb_face_node = 0;
  Integer nb_node_x  = nb_cell_x+1;
  Integer nb_node_y  = -1;
  Integer nb_node_z  = nb_cell_z+1;

  if (m_mesh_dimension==3){
    nb_node_y = nb_cell_y + 1;
  }
  else if (m_mesh_dimension==2){
    nb_node_y = 1;
  }
  else if (m_mesh_dimension==1){
    nb_node_y = 1;
    nb_node_z = 1;
  }

  Integer nb_node_xy = CheckedConvert::multiply(nb_node_x,nb_node_y);
  Integer nb_node_yz = CheckedConvert::multiply(nb_node_y,nb_node_z);

  Integer nb_cell_xy = CheckedConvert::multiply(nb_cell_x,nb_cell_y);
  Integer nb_cell_yz = CheckedConvert::multiply(nb_cell_y,nb_cell_z);
  Integer nb_cell_xz = CheckedConvert::multiply(nb_cell_x,nb_cell_z);
  
  Integer nb_face_x  = CheckedConvert::multiply(nb_node_x,nb_cell_yz);
  Integer nb_face_y  = CheckedConvert::multiply(nb_node_y,nb_cell_xz);
  Integer nb_face_z  = CheckedConvert::multiply(nb_node_z,nb_cell_xy);

  Integer mesh_nb_cell = CheckedConvert::multiply(nb_cell_x,nb_cell_y,nb_cell_z);
  Integer mesh_nb_node = CheckedConvert::multiply(nb_node_x,nb_node_y,nb_node_z);
  Integer mesh_nb_face = nb_face_x + nb_face_y + nb_face_z;

  // Pour tester des uid>32bits, il suffit
  // de changer ce multiplier et de le mettre
  // par exemple a 1000000
  const Int64 uid_multiplier = 1;
  //const Int64 uid_multiplier = 12500000;
  
  info() << "mesh generation dim=" << m_mesh_dimension;
  info() << "First mesh layer: " << first_cell_z;
  info() << "Number of cells layers: " << nb_cell_z << '/' << total_para_cell_z;
  info() << "First node layer: " << first_node_z;
  info() << "Number of nodes layers: " << nb_node_z << '/' << total_para_node_z;

  info() << "Number of nodes  " << mesh_nb_node;
  info() << "Number of faces    " << mesh_nb_face;
  info() << "Number of cells  " << mesh_nb_cell;

  info() << "nb node_yz=" << nb_node_yz;

  // Création des noeuds

  Int64UniqueArray nodes_unique_id(mesh_nb_node);

  HashTableMapT<Int64,NodeInfo> nodes_infos(mesh_nb_node,true);
  
  Real ydelta = (m_xyz_delta[1]==0.0) ? ARCANE_REAL(0.02) : m_xyz_delta[1];
  Real zdelta = (m_xyz_delta[2]==0.0) ? ARCANE_REAL(0.02) : m_xyz_delta[2];
  Real xdelta = (m_xyz_delta[0]==0.0) ? ARCANE_REAL(1.0)/(Real)(nb_cell_x):m_xyz_delta[0];
  // Le milieu pour determiner ZG et ZD
  Real middle_x = ARCANE_REAL(0.5);
  // Le milieu pour determiner HAUT et BAS
  Real middle_height = ARCANE_REAL((nb_cell_y/2)*ydelta);
  
  if (m_xyz_delta[0]!=0.0)
    middle_x = m_xyz_delta[0] * (nb_cell_x/2);
  
  if (m_mesh_dimension==2){
    zdelta = ydelta;
    middle_height = (nb_cell_z/2)*zdelta;
  }

  info() << "Xdelta=" << xdelta<< ", Ydelta=" << ydelta << ", Zdelta=" << zdelta;
  info() << "middle_x=" << middle_x<< ", middle_height=" << middle_height;

  // Création des noeuds
  Integer nb_node_local_id = 0;
  if (m_zyx_generate==false){
    Integer node_local_id = 0;
    for( Integer x=0; x<nb_node_x; ++x ){
      for( Integer z=0; z<nb_node_z; ++z ){
				for( Integer y=0; y<nb_node_y; ++y ){
	  
					Real nx = xdelta * (Real)(x);
					Real ny = ydelta * (Real)(y);
					Real nz = zdelta * (Real)(z+first_node_z);
          if (m_mesh_dimension==2){
            ny = nz;
            nz = 0.0;
          }
          else if (m_mesh_dimension==1){
            ny = nz = 0.0;
          }
          Int64 node_unique_id = (Int64)y + ((Int64)z+first_node_z)*(Int64)nb_node_y + (Int64)x*nb_node_y*total_para_node_z;
          node_unique_id *= uid_multiplier;
					
          nodes_unique_id[node_local_id] = node_unique_id;
          Integer owner = my_part;
          // S'il s'agit de la couche de noeud du dessus (z max),
          // elle appartient au sous-domaine suivant
          if (z==(nb_node_z-1) && !is_last_proc)
            owner = my_part+1;

          nodes_infos.nocheckAdd(node_unique_id,NodeInfo(owner,Real3(nx,ny,nz)));

          //debug(Trace::High) << "Add coord uid=" << node_unique_id << " pos=" << Real3(nx,ny,nz);
          //info() << "Add coord uid=" << node_unique_id << " pos=" << Real3(nx,ny,nz);
          
          ++node_local_id;
        }
      }
    }
    nb_node_local_id = node_local_id;
  }
  else{ // m_zyx_generate
    Integer node_local_id = 0;
    for(Integer z=0; z<nb_node_z; ++z ){
      for(Integer y=0; y<nb_node_y; ++y ){
        for(Integer x=0; x<nb_node_x; ++x ){
          Real nx = xdelta * (Real)(x);
          Real ny = ydelta * (Real)(y);
          Real nz = zdelta * (Real)(z+first_node_z);
          if (m_mesh_dimension==2){
            ny = nz;
            nz = 0.0;
          } else if (m_mesh_dimension==1){
            ny = nz = 0.0;
          }
          Int64 node_unique_id = (Int64)x + (Int64)y*nb_node_x + ((Int64)z+first_node_z)*(Int64)nb_node_x*(Int64)nb_node_y;
          node_unique_id *= uid_multiplier;
          
          nodes_unique_id[node_local_id] = node_unique_id;
          Integer owner = my_part;
          // S'il s'agit de la couche de noeud du dessus (z max),
          // elle appartient au sous-domaine suivant
          if (z==(nb_node_z-1) && !is_last_proc) owner = my_part+1;
          nodes_infos.nocheckAdd(node_unique_id,NodeInfo(owner,Real3(nx,ny,nz)));
          //debug(Trace::High) << "Add coord uid=" << node_unique_id << " pos=" << Real3(nx,ny,nz);
          //info() << "Add coord uid=" << node_unique_id << " pos=" << Real3(nx,ny,nz);
          //info() << "[SodMeshGenerator::generateMesh] node @ "<<x<<"x"<<y<<"x"<<z<<":"<<", uid=" << node_unique_id;
         ++node_local_id;
        }
      }
    }
    nb_node_local_id = node_local_id;
  }

  // Création des mailles

  // Infos pour la création des mailles
  // par maille: 1 pour son unique id,
  //             1 pour son type,
  //             8 pour chaque noeud
  Int64UniqueArray cells_infos(mesh_nb_cell*10);

  if (m_mesh_dimension==1){
    Integer cells_infos_index = 0;
    
    for( Integer x=0; x<nb_cell_x; ++x ){
      for( Integer z=0; z<nb_cell_z; ++z ){
        for( Integer y=0; y<nb_cell_y; ++y ){
          Integer current_cell_nb_node = 2;
          
          Int64 cell_unique_id = _generateCellUniqueId(x,y,z,nb_cell_y,first_cell_z,total_para_cell_z);
          cell_unique_id *= uid_multiplier;

          cells_infos[cells_infos_index] = IT_CellLine2;
          ++cells_infos_index;

          cells_infos[cells_infos_index] = cell_unique_id;
          ++cells_infos_index;

          Integer base_id = y + z*nb_node_y + x*nb_node_yz;
          cells_infos[cells_infos_index+0] = nodes_unique_id[base_id];
          cells_infos[cells_infos_index+1] = nodes_unique_id[base_id + 1];

          cells_infos_index += current_cell_nb_node;
        }
      }
    }
  }
  else if (m_mesh_dimension==2){
    Integer cells_infos_index = 0;

    for( Integer x=0; x<nb_cell_x; ++x ){
      for( Integer z=0; z<nb_cell_z; ++z ){
        for( Integer y=0; y<nb_cell_y; ++y ){
          Integer current_cell_nb_node = 4;
          
          Int64 cell_unique_id = _generateCellUniqueId(x,y,z,nb_cell_y,first_cell_z,total_para_cell_z);
          cell_unique_id *= uid_multiplier;

          cells_infos[cells_infos_index] = IT_Quad4;
          ++cells_infos_index;

          cells_infos[cells_infos_index] = cell_unique_id;
          ++cells_infos_index;

          Integer base_id = y + z*nb_node_y + x*nb_node_yz;
          cells_infos[cells_infos_index+0] = nodes_unique_id[base_id];
          cells_infos[cells_infos_index+1] = nodes_unique_id[base_id + nb_node_yz];
          cells_infos[cells_infos_index+2] = nodes_unique_id[base_id + nb_node_yz + 1];
          cells_infos[cells_infos_index+3] = nodes_unique_id[base_id + 1];
          
          cells_infos_index += current_cell_nb_node;
        }
      }
    }
  }
  else if (m_mesh_dimension==3){
    if (m_zyx_generate==false){
      Integer cells_infos_index = 0;
      for( Integer x=0; x<nb_cell_x; ++x ){
        for( Integer z=0; z<nb_cell_z; ++z ){
          for( Integer y=0; y<nb_cell_y; ++y ){
            Integer current_cell_nb_node = 8;
            Int64 cell_unique_id = _generateCellUniqueId(x,y,z,nb_cell_y,first_cell_z,total_para_cell_z);
            cell_unique_id *= uid_multiplier;
//          info() << "[SodMeshGenerator::generateMesh] "<<x<<"x"<<y<<"x"<<z<<":"<<", uid=" << cell_unique_id;
            cells_infos[cells_infos_index] = IT_Hexaedron8;
            ++cells_infos_index;
            cells_infos[cells_infos_index] = cell_unique_id;
            ++cells_infos_index;
            Integer base_id = y + z*nb_node_y + x*nb_node_yz;
            cells_infos[cells_infos_index+0] = nodes_unique_id[base_id];
            cells_infos[cells_infos_index+1] = nodes_unique_id[base_id + 1];
            cells_infos[cells_infos_index+2] = nodes_unique_id[base_id + nb_node_y + 1];
            cells_infos[cells_infos_index+3] = nodes_unique_id[base_id + nb_node_y + 0];
            cells_infos[cells_infos_index+4] = nodes_unique_id[base_id + nb_node_yz];
            cells_infos[cells_infos_index+5] = nodes_unique_id[base_id + nb_node_yz + 1];
            cells_infos[cells_infos_index+6] = nodes_unique_id[base_id + nb_node_yz + nb_node_y + 1];
            cells_infos[cells_infos_index+7] = nodes_unique_id[base_id + nb_node_yz + nb_node_y + 0];
            cells_infos_index += current_cell_nb_node;
          }
        }
      }
    }else{ // m_zyx_generate
      Integer cells_infos_index = 0;
      for( Integer z=0; z<nb_cell_z; ++z ){
        for( Integer y=0; y<nb_cell_y; ++y ){
          for( Integer x=0; x<nb_cell_x; ++x ){
            Integer current_cell_nb_node = 8;
            Int64 cell_unique_id = _generateCellUniqueId(x,y,z,nb_cell_y,first_cell_z,total_para_cell_z);
            cell_unique_id *= uid_multiplier;
            debug() << "[SodMeshGenerator::generateMesh] + m_zyx_generate "<<x<<"x"<<y<<"x"<<z<<":"<<", uid=" << cell_unique_id;
            cells_infos[cells_infos_index] = IT_Hexaedron8;
            ++cells_infos_index;
            cells_infos[cells_infos_index] = cell_unique_id;
            ++cells_infos_index;
            Integer base_id = x + y*nb_node_x + z*nb_node_xy;
            cells_infos[cells_infos_index+0] = nodes_unique_id[base_id];
            cells_infos[cells_infos_index+1] = nodes_unique_id[base_id                          + 1];
            cells_infos[cells_infos_index+2] = nodes_unique_id[base_id              + nb_node_x + 1];
            cells_infos[cells_infos_index+3] = nodes_unique_id[base_id              + nb_node_x + 0];
            cells_infos[cells_infos_index+4] = nodes_unique_id[base_id + nb_node_xy];
            cells_infos[cells_infos_index+5] = nodes_unique_id[base_id + nb_node_xy             + 1];
            cells_infos[cells_infos_index+6] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 1];
            cells_infos[cells_infos_index+7] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 0];
            debug() << "[SodMeshGenerator::generateMesh] + m_zyx_generate nodes: "
                   << cells_infos[cells_infos_index+0] << ", "
                   << cells_infos[cells_infos_index+1] << ", "
                   << cells_infos[cells_infos_index+2] << ", "
                   << cells_infos[cells_infos_index+3] << ", "
                   << cells_infos[cells_infos_index+4] << ", "
                   << cells_infos[cells_infos_index+5] << ", "
                   << cells_infos[cells_infos_index+6] << ", "
                   << cells_infos[cells_infos_index+7];
            cells_infos_index += current_cell_nb_node;
          }
        }
      }
    }
  }

  mesh->setDimension(m_mesh_dimension);
  mesh->allocateCells(mesh_nb_cell,cells_infos,true);
  
  VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
  {
    // Remplit la variable contenant les coordonnées des noeuds
    Int32UniqueArray nodes_local_id(nodes_unique_id.size());
    IItemFamily* family = mesh->itemFamily(IK_Node);
    family->itemsUniqueIdToLocalId(nodes_local_id,nodes_unique_id);
    NodeInfoListView nodes_internal(family);
    for( Integer i=0; i<nb_node_local_id; ++i ){
      Node node = nodes_internal[nodes_local_id[i]];
      Int64 unique_id = nodes_unique_id[i];
      nodes_coord_var[node] = nodes_infos.lookupValue(unique_id).m_coord;
      //info() << "Set coord " << ItemPrinter(node) << " coord=" << nodes_coord_var[node]
      //       << " coord2=" << nodes_infos.lookupValue(unique_id).m_coord;
    }
  }
  nodes_coord_var.synchronize();

  {
    SodStandardGroupsBuilder groups_builder(traceMng());
    Real max_x = xdelta * nb_cell_x;
    Real max_y = ydelta * nb_cell_y;
    // Dans le cas 2D, nb_cell_z et nb_cell_y sont swappés
    if (m_mesh_dimension==2){
      max_y = ydelta * Convert::toReal(total_para_cell_z);
      info()<< "[SodMeshGenerator::generateMesh]  max_y=" << max_y;
    }
    const Real max_z = zdelta * Convert::toReal(total_para_cell_z);
    groups_builder.generateGroups(mesh, Real3::null(), Real3(max_x, max_y, max_z), middle_x, middle_height, true);
  }

  bool is_random = !math::isNearlyZero(m_random_coef);
  if (is_random){
    if (m_mesh_dimension==1)
      throw NotImplementedException(A_FUNCINFO,"Randomisation for 1D mesh");
    info() << "** ** ** Randomize node positions coef=" << m_random_coef;
    for( Integer x=0; x<nb_node_x; ++x ){
      for( Integer z=0; z<nb_node_z; ++z ){
				for( Integer y=0; y<nb_node_y; ++y ){
          Real xd = xdelta;
          Real yd = ydelta;
          Real zd = zdelta;
          if (x!=0 && (x+1)!=nb_node_x && (z+first_node_z)!=0
              && (z+first_node_z)!=(total_para_node_z-1)
              && (m_mesh_dimension==2 || (y!=0 && y!=(nb_node_y-1)))){
            Real xr = (Real)::rand() / (Real)RAND_MAX;
            Real yr = (Real)::rand() / (Real)RAND_MAX;
            Real zr = (Real)::rand() / (Real)RAND_MAX;
            //info() << " xr=" << xr << "yr=" << yr << " zr=" << zr;
            xd = xd + (xr-0.5)*xdelta*m_random_coef;
            yd = yd + (yr-0.5)*ydelta*m_random_coef;
            zd = zd + (zr-0.5)*zdelta*m_random_coef;
          }

          if (m_mesh_dimension==2){
            yd = zd;
            zd = 0.0;
          }

          Int64 node_unique_id = y + (z+first_node_z)*nb_node_y + x*nb_node_y*total_para_node_z;
          Real3 pos = nodes_infos.lookupValue(node_unique_id).m_coord;

          pos += Real3(xd,yd,zd);

          nodes_infos.add(node_unique_id,NodeInfo(0,pos));
        }
      }
    }
    ENUMERATE_NODE(inode,mesh->ownNodes()){
      Node node = *inode;
      nodes_coord_var[inode] = nodes_infos.lookupValue(node.uniqueId()).m_coord;
    }
    nodes_coord_var.synchronize();
  }
  info() << "End of mesh generation";
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de génération d'un tube à choc en 3D.
 */
class Sod3DMeshGenerator
: public ArcaneSod3DMeshGeneratorObject
{
 public:
  Sod3DMeshGenerator(const ServiceBuildInfo& sbi)
  : ArcaneSod3DMeshGeneratorObject(sbi){}
 public:
  void fillMeshBuildInfo(MeshBuildInfo& build_info) override
  {
    ARCANE_UNUSED(build_info);
  }
  void allocateMeshItems(IPrimaryMesh* pm) override
  {
    bool zyx = false;
    Int32 wanted_x = options()->x();
    Int32 wanted_y = options()->y();
    Int32 wanted_z = options()->z();
    Real random_coef = 0.0;
    Real delta0 = 0.0;
    Real delta1 = 0.0;
    Real delta2 = 0.0;
    bool z_is_total = false;
    SodMeshGenerator::Impl x(traceMng(),zyx,wanted_x,wanted_y,wanted_z,
                             z_is_total,3,random_coef,delta0,delta1,delta2);
    x.generateMesh(pm);
  }
};

ARCANE_REGISTER_SERVICE_SOD3DMESHGENERATOR(Sod3D,Sod3DMeshGenerator);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
