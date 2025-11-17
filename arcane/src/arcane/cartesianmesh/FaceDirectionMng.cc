// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceDirectionMng.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Infos sur les faces d'une direction X Y ou Z d'un maillage structuré.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/FaceDirectionMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"
#include "arcane/VariableTypes.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FaceDirectionMng::Impl
{
 public:
  Impl() : m_infos(platform::getDefaultDataAllocator()){}
 public:
  FaceGroup m_inner_all_items;
  FaceGroup m_outer_all_items;
  FaceGroup m_all_items;
  ICartesianMesh* m_cartesian_mesh = nullptr;
  Integer m_patch_index = -1;
  UniqueArray<ItemDirectionInfo> m_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceDirectionMng::
FaceDirectionMng()
: m_direction(MD_DirInvalid)
, m_p (nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceDirectionMng::
_internalInit(ICartesianMesh* cm,eMeshDirection dir,Integer patch_index)
{
  if (m_p)
    ARCANE_FATAL("Initialisation already done");
  m_p = new Impl();
  m_direction = dir;
  m_p->m_cartesian_mesh = cm;
  m_p->m_patch_index = patch_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceDirectionMng::
_internalDestroy()
{
  delete m_p;
  m_p = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceDirectionMng::
_internalResizeInfos(Int32 new_size)
{
  m_p->m_infos.resize(new_size);
  m_infos_view = m_p->m_infos.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceDirectionMng::
_internalComputeInfos(const CellDirectionMng& cell_dm,const VariableCellReal3& cells_center,
                      const VariableFaceReal3& faces_center)
{
  IMesh* mesh = m_p->m_cartesian_mesh->mesh();
  IItemFamily* face_family = mesh->faceFamily();
  IItemFamily* cell_family = mesh->cellFamily();
  int dir = (int)m_direction;
  String base_group_name = String("Direction")+dir;
  if (m_p->m_patch_index>=0)
    base_group_name = base_group_name + String("AMRPatch")+m_p->m_patch_index;

  // Calcule la liste des faces dans une direction donnée.
  // Il faut pour chaque maille ajouter dans la liste des faces
  // les deux faces de la direction souhaitées en prenant bien soin
  // de ne pas ajouter deux fois la même face.
  UniqueArray<Int32> faces_lid;
  {
    CellGroup all_cells = cell_dm.allCells();
    faces_lid.reserve(all_cells.size());
    // Ensemble des faces déjà ajoutées
    std::set<Int32> done_faces;
    ENUMERATE_CELL(icell,all_cells){
      DirCellFace dcf(cell_dm.cellFace(*icell));
      Face next_face = dcf.next();
      Face prev_face = dcf.previous();

      //! Ajoute la face d'avant à la liste des faces de cette direction
      Int32 prev_lid = prev_face.localId();
      if (done_faces.find(prev_lid)==done_faces.end()){
        faces_lid.add(prev_lid);
        done_faces.insert(prev_lid);
      }
      Int32 next_lid = next_face.localId();
      if (done_faces.find(next_lid)==done_faces.end()){
        faces_lid.add(next_lid);
        done_faces.insert(next_lid);
      }
    }
  }

  FaceGroup all_faces = face_family->createGroup(String("AllFaces")+base_group_name,Int32ConstArrayView(),true);
  all_faces.setItems(faces_lid,true);

  UniqueArray<Int32> inner_cells_lid;
  cell_dm.innerCells().view().fillLocalIds(inner_cells_lid);

  UniqueArray<Int32> inner_lids;
  UniqueArray<Int32> outer_lids;
  ENUMERATE_ (Face, iface, all_faces) {
    Int32 lid = iface.itemLocalId();
    Face face = *iface;
    // TODO: ne pas utiser nbCell() mais faire cela via le std::set utilisé précédemment
    if (face.nbCell() == 1) {
      outer_lids.add(lid);
    }
    else {
      bool c0_inner_cell = inner_cells_lid.contains(face.cell(0).localId());
      bool c1_inner_cell = inner_cells_lid.contains(face.cell(1).localId());
      // Si au moins une des mailles est interne, alors la face est interne.
      // TODO : Gérer les faces communes.
      if (c0_inner_cell || c1_inner_cell) {
        inner_lids.add(lid);
      }
      else {
        outer_lids.add(lid);
      }
    }
  }
  m_p->m_inner_all_items = face_family->createGroup(String("AllInner")+base_group_name,inner_lids,true);
  m_p->m_outer_all_items = face_family->createGroup(String("AllOuter")+base_group_name,outer_lids,true);
  m_p->m_all_items = all_faces;
  m_cells = CellInfoListView(cell_family);

  _computeCellInfos(cell_dm,cells_center,faces_center);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool FaceDirectionMng::
_hasFace(Cell cell,Int32 face_local_id) const
{
  for( FaceLocalId iface_lid : cell.faceIds() ){
    if (iface_lid==face_local_id)
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule des mailles avant et après une face, dans une direction donnée.
 *
 * Pour être indépendant de la façon dont est créé le maillage, on utilise les coordonnées
 * des centres des faces et des centres des mailles.
 */
void FaceDirectionMng::
_computeCellInfos(const CellDirectionMng& cell_dm,const VariableCellReal3& cells_center,
                  const VariableFaceReal3& faces_center)
{
  eMeshDirection dir = m_direction;

  // Créé l'ensemble des mailles du patch et s'en sert
  // pour être sur que chaque maille devant/derrière est dans
  // cet ensemble
  std::set<Int32> patch_cells_set;
  ENUMERATE_CELL(icell,cell_dm.allCells()){
    patch_cells_set.insert(icell.itemLocalId());
  }

  ENUMERATE_FACE(iface,m_p->m_all_items){
    Face face = *iface;
    Int32 face_lid = iface.itemLocalId();
    Real3 face_coord = faces_center[iface];
    Cell front_cell = face.frontCell();
    Cell back_cell = face.backCell();

    // Vérifie que les mailles sont dans notre patch.
    // if (!front_cell.null())
    //   if (patch_cells_set.find(front_cell.localId())==patch_cells_set.end())
    //     front_cell = Cell();
    // if (!back_cell.null())
    //   if (patch_cells_set.find(back_cell.localId())==patch_cells_set.end())
    //     back_cell = Cell();

    bool is_inverse = false;
    if (!front_cell.null()){
      Real3 front_coord = cells_center[front_cell];
      if (dir==MD_DirX){
        if (front_coord.x<face_coord.x)
          is_inverse = true;
      }
      else if (dir==MD_DirY){
        if (front_coord.y<face_coord.y)
          is_inverse = true;
      }
      else if (dir==MD_DirZ){
        if (front_coord.z<face_coord.z)
          is_inverse = true;
      }
    }
    else{
      Real3 back_coord = cells_center[back_cell];
      if (dir==MD_DirX){
        if (back_coord.x>face_coord.x)
          is_inverse = true;
      }
      else if (dir==MD_DirY){
        if (back_coord.y>face_coord.y)
          is_inverse = true;
      }
      else if (dir==MD_DirZ){
        if (back_coord.z>face_coord.z)
          is_inverse = true;
      }
    }
    // Si la face a deux mailles connectées, regarde le niveau AMR de ces
    // deux mailles et s'il est différent, ne conserve que la maille
    // dont le niveau AMR est celui de la face.
    if (!back_cell.null() && !front_cell.null()){
      Int32 back_level = back_cell.level();
      Int32 front_level = front_cell.level();
      if (back_level!=front_level){
        // La face n'a pas l'information de son niveau mais si les deux
        // mailles ne sont pas de même niveau la face n'appartient qu'à une
        // seule des deux mailles. On ne garde donc que cette dernière.
        if (!_hasFace(back_cell,face_lid))
          back_cell = Cell();
        if (!_hasFace(front_cell,face_lid))
          front_cell = Cell();
      }
    }
    if (is_inverse)
      m_infos_view[face_lid] = ItemDirectionInfo(back_cell, front_cell);
    else
      m_infos_view[face_lid] = ItemDirectionInfo(front_cell, back_cell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup FaceDirectionMng::
allFaces() const
{
  return m_p->m_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup FaceDirectionMng::
innerFaces() const
{
  return m_p->m_inner_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup FaceDirectionMng::
outerFaces() const
{
  return m_p->m_outer_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
