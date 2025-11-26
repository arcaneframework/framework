// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SodStandardGroupsBuilder.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Création des groupes pour les cas test de tube à choc de Sod.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/SodStandardGroupsBuilder.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Real3.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/Item.h"
#include "arcane/ItemPrinter.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/MeshVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SodStandardGroupsBuilder::
_createFaceGroup(IMesh* mesh,const String& name, Int32ConstArrayView faces_lid)
{
  info() << "Creation group de face '" << name << "'"
         << " size=" << faces_lid.size();

  IItemFamily* face_family = mesh->faceFamily();
  if (arcaneIsDebug()){
    FaceInfoListView mesh_faces(face_family);
    VariableNodeReal3& var_nodes(mesh->nodesCoordinates());
    for( Integer z=0, zs=faces_lid.size(); z<zs; ++ z){
      Face face(mesh_faces[faces_lid[z]]);
      debug(Trace::High) << "Face " << ItemPrinter(face);
      for( NodeLocalId inode : face.nodes() )
        debug(Trace::Highest) << "XYZ= " << var_nodes[inode];
    }
  }

  face_family->createGroup(name,faces_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SodStandardGroupsBuilder::
generateGroups(IMesh* mesh, Real3 min_pos, Real3 max_pos, Real middle_x, Real middle_height, bool do_zg_and_zd)
{
  VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
  Int32 mesh_dimension = mesh->dimension();

  const Real min_x = min_pos.x;
  const Real min_y = min_pos.y;
  const Real min_z = min_pos.z;
  const Real max_x = max_pos.x;
  const Real max_y = max_pos.y;
  const Real max_z = max_pos.z;
  IItemFamily* cell_family = mesh->cellFamily();
  info() << "Generate standard groups for SOD: "
         << " min=" << min_pos << " max=" << max_pos
         << " middle_x=" << middle_x << " middle_height=" << middle_height;

  {
    UniqueArray<Int32> xmin_surface_lid;
    UniqueArray<Int32> xmax_surface_lid;
    UniqueArray<Int32> ymin_surface_lid;
    UniqueArray<Int32> ymax_surface_lid;
    UniqueArray<Int32> zmin_surface_lid;
    UniqueArray<Int32> zmax_surface_lid;

    ENUMERATE_ (Face, iface, mesh->allFaces()) {
      Face face = *iface;
      Int32 face_local_id = face.localId();
      bool is_xmin = true;
      bool is_xmax = true;
      bool is_ymin = true;
      bool is_ymax = true;
      bool is_zmin = true;
      bool is_zmax = true;
      //info() << "Face local id = " << face_local_id;
      for( Node inode : face.nodes() ){
        Real3 coord = nodes_coord_var[inode];
        if (!math::isNearlyEqual(coord.x,min_x))
          is_xmin = false;
        if (!math::isNearlyEqual(coord.x,max_x))
          is_xmax = false;

        if (!math::isNearlyEqual(coord.y,min_y))
          is_ymin = false;
        if (!math::isNearlyEqual(coord.y,max_y))
          is_ymax = false;
        
        if (!math::isNearlyEqual(coord.z,min_z))
          is_zmin = false;
        if (!math::isNearlyEqual(coord.z,max_z))
          is_zmax = false;
      }
      if (is_xmin)
        xmin_surface_lid.add(face_local_id);
      if (is_xmax)
        xmax_surface_lid.add(face_local_id);
      if (is_ymin)
        ymin_surface_lid.add(face_local_id);
      if (is_ymax)
        ymax_surface_lid.add(face_local_id);
      if (is_zmin)
        zmin_surface_lid.add(face_local_id);
      if (is_zmax)
        zmax_surface_lid.add(face_local_id);
      
    }
    _createFaceGroup(mesh,"XMIN",xmin_surface_lid);
    _createFaceGroup(mesh,"XMAX",xmax_surface_lid);
    if (mesh_dimension>=2){
      _createFaceGroup(mesh,"YMIN",ymin_surface_lid);
      _createFaceGroup(mesh,"YMAX",ymax_surface_lid);
    }
    if (mesh_dimension==3){
      _createFaceGroup(mesh,"ZMIN",zmin_surface_lid);
      _createFaceGroup(mesh,"ZMAX",zmax_surface_lid);
    }
  }

  // Détermine les couches ZG et ZD
  if (do_zg_and_zd) {
    UniqueArray<Int32> zg_lid;
    UniqueArray<Int32> zd_lid;
    const Real xlimit = middle_x; // Position séparant les deux zones ZG et ZD
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      bool is_in_zd = false;
      bool is_in_zg = false;
      const Cell& cell = *icell;
      Integer local_id = cell.localId();
      for( Node inode : cell.nodes() ){
        Real x = nodes_coord_var[inode].x;
        if (math::isNearlyEqual(x,xlimit))
          continue;
        if (x > xlimit)
          is_in_zd = true;
        else
          is_in_zg = true;
      }
      if (is_in_zg){
        if (is_in_zd){
          for( Node inode : cell.nodes() ){
            Real x = nodes_coord_var[inode].x;
            info() << " X=" << x;
            if (math::isNearlyEqual(x,xlimit))
              info() << " is equal to " << xlimit;
            {
              Real s = math::abs(x) + math::abs(xlimit);
              Real d = x - xlimit;
              Real r = d/s;
              info() << " nearly s=" << s << " d=" << d << " r=" << r;
              info() << " r<0.0=" << (r<0.0) << " r>-e=" << (r>-FloatInfo<Real>::nearlyEpsilon())
                     << " r<e=" << (r<FloatInfo<Real>::nearlyEpsilon());
            }
            if (x > xlimit)
              info() << " is in zd";
            else
              info() << " is in zg";
          }
          ARCANE_FATAL("SodGenerator: cell '{0}' is in both ZG and ZD",ItemPrinter(cell));
        }
        debug(Trace::High) << "Add Cell " << ItemPrinter(cell) << " to ZG";
        zg_lid.add(local_id);
      }
      else{
        debug(Trace::High) << "Add Cell " << ItemPrinter(cell) << " to ZD";
        zd_lid.add(local_id);
      }
    }
    info() << "Create ZG with " << zg_lid.size() << " cells";
    cell_family->createGroup("ZG",zg_lid);
    info() << "Create ZD with " << zd_lid.size() << " cells";
    cell_family->createGroup("ZD",zd_lid);
  }
  
  // Détermine les groupes ZD_HAUT et ZD_BAS
  if (do_zg_and_zd) {
    ItemGroup zdGroup = cell_family->findGroup("ZD");
    if (zdGroup.null())
      ARCANE_FATAL("Group 'ZD' has not been found!");
    UniqueArray<Int32> zd_bas_lid;
    UniqueArray<Int32> zd_haut_lid;
    const Real height_limit = middle_height; // Position séparant les deux zones HAUT et BAS
    ENUMERATE_CELL(icell, zdGroup){
      bool is_in_zd_bas = false;
      bool is_in_zd_haut = false;
      Cell cell = *icell;
      Integer local_id = cell.localId();
      for( Node inode : cell.nodes() ){
        Real height = (mesh_dimension==2)?nodes_coord_var[inode].y:nodes_coord_var[inode].z;
        //info()<< "\tcoords="<<nodes_coord_var[inode]<<", height="<<height<<" vs "<<height_limit;
        if (math::isNearlyEqual(height, height_limit))
          continue;
        if (height > height_limit)
          is_in_zd_haut = true;
        else
          is_in_zd_bas = true;
      }
      if (is_in_zd_bas){
        if (is_in_zd_haut)
          fatal() << "SodGenerator: cell " << ItemPrinter(cell) << " in ZD_BAS and ZD_HAUT";
        debug(Trace::High) << "Add Cell " << ItemPrinter(cell) << " to ZD_BAS";
        zd_bas_lid.add(local_id);
      }
      else{
        debug(Trace::High) << "Add Cell " << ItemPrinter(cell) << " to ZD_HAUT";
        zd_haut_lid.add(local_id);
      }
    }
    info() << "Create ZD_BAS with " << zd_bas_lid.size() << " cells";
    cell_family->createGroup("ZD_BAS",zd_bas_lid);
    info() << "Create ZD_HAUT with " << zd_haut_lid.size() << " cells";
    cell_family->createGroup("ZD_HAUT",zd_haut_lid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
