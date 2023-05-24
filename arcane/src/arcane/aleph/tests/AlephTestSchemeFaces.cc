// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "arcane/aleph/tests/AlephTest.h"
#include "arcane/aleph/tests/AlephTestScheme.h"
#include "arcane/aleph/tests/AlephTestSchemeFaces.h"

#include "arcane/mesh/CellFamily.h"
#include "arcane/mesh/FaceFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

ARCANE_REGISTER_SERVICE_ALEPHTESTSCHEMEFACES(Faces, AlephTestSchemeFaces);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephTestSchemeFaces::
AlephTestSchemeFaces(const ServiceBuildInfo& sbi)
: ArcaneAlephTestSchemeFacesObject(sbi)
{
  debug() << "\33[37m\t[AlephTestSchemeFaces::AlephTestSchemeFaces] New"
          << "\33[0m";
}

AlephTestSchemeFaces::~AlephTestSchemeFaces(void)
{
  debug() << "\33[37m\t[AlephTestSchemeFaces::AlephTestSchemeFaces] Delete"
          << "\33[0m";
}

/***************************************************************************
 * Application des conditions aux limites propres au schéma
 ***************************************************************************/
void AlephTestSchemeFaces::
boundaries(ArcaneTest::CaseOptionsAlephTestModule* module_options)
{
  ItacFunction(AlephTestSchemeFaces);
  // boucle sur les conditions aux limites
  for (int i = module_options->boundaryCondition.size() - 1; i >= 0; --i) {
    Real temperature = module_options->boundaryCondition[i]->value();
    FaceGroup face_group = module_options->boundaryCondition[i]->surface();
    // boucle sur les faces de la surface
    ENUMERATE_FACE (iFace, face_group) {
      m_face_temperature[iFace] = temperature;
    }
  }
}

/***************************************************************************
 * On compte le nombre d'éléments non nuls par ligne de la matrice
 ***************************************************************************/
void AlephTestSchemeFaces::
preFetchNumElementsForEachRow(IntegerArray& rows_nb_element,
                              const Integer rank_row_offset)
{
  ItacFunction(AlephTestSchemeFaces);
  debug() << "\33[37m\t[AlephTestSchemeFaces::preFetchNumElementsForEachRow]"
          << "\33[0m";
  // On comptabilise les termes de la diagonale
  rows_nb_element.fill(1);

  // Et on rajoute les couplages par faces
  // Attention, il faut bien parcourir toutes les mailles et filtrer les faces des fantômes
  ENUMERATE_FACE (iFace, INNER_ACTIVE_FACE_GROUP(allCells())) {
    if (iFace->backCell().isOwn())
      rows_nb_element[m_cell_matrix_idx[iFace->backCell()] - rank_row_offset] += 1;
    if (iFace->frontCell().isOwn())
      rows_nb_element[m_cell_matrix_idx[iFace->frontCell()] - rank_row_offset] += 1;
  }
}

/***************************************************************************
 * AlephTestModule::_FaceComputeSetValues                                  *
 ***************************************************************************/
void AlephTestSchemeFaces::
setValues(const Real deltaT, AlephMatrix* aleph_mat)
{
  // On flush les coefs
  ENUMERATE_CELL (iCell, MESH_OWN_ACTIVE_CELLS(mesh()))
    m_cell_coefs[iCell] = 0.;
  // Faces 'inner'
  debug() << "\33[37m[AlephTestSchemeFaces::setValues] inner-faces"
          << "\33[0m";
  ENUMERATE_FACE (iFace, INNER_ACTIVE_FACE_GROUP(allCells())) {
    if (iFace->backCell().isOwn()) {
      const Real surface = AlephTestModule::geoFaceSurface(*iFace, nodesCoordinates());
      aleph_mat->setValue(m_cell_matrix_idx[iFace->backCell()],
                          m_cell_matrix_idx[iFace->frontCell()], -deltaT / surface);
      m_cell_coefs[iFace->backCell()] += 1.0 / surface;
    }
    if (iFace->frontCell().isOwn()) {
      const Real surface = AlephTestModule::geoFaceSurface(*iFace, nodesCoordinates());
      aleph_mat->setValue(m_cell_matrix_idx[iFace->frontCell()],
                          m_cell_matrix_idx[iFace->backCell()], -deltaT / surface);
      m_cell_coefs[iFace->frontCell()] += 1.0 / surface;
    }
  }
  debug() << "\33[37m[AlephTestSchemeFaces::setValues] outer-faces"
          << "\33[0m";
  ENUMERATE_FACE (iFace, OUTER_ACTIVE_FACE_GROUP(allCells())) {
    if (!iFace->cell(0).isOwn())
      continue;
    m_cell_coefs[iFace->cell(0)] += 1.0 / AlephTestModule::geoFaceSurface(*iFace, nodesCoordinates());
  }
  debug() << "\33[37m[AlephTestSchemeFaces::setValues] diagonale"
          << "\33[0m";
  ENUMERATE_CELL (iCell, MESH_OWN_ACTIVE_CELLS(mesh())) {
    aleph_mat->setValue(m_cell_matrix_idx[iCell], m_cell_matrix_idx[iCell], 1.0 + deltaT * m_cell_coefs[iCell]);
  }
  debug() << "\33[37m[AlephTestSchemeFaces::setValues] done"
          << "\33[0m";
}

/***************************************************************************
 * AlephTestModule::_amrRefine *
 ***************************************************************************/
bool AlephTestSchemeFaces::
amrRefine(RealArray& values, const Real trigRefine)
{
  ItacFunction(AlephTestSchemeFaces);
  if (!options()->amr)
    return false;
  debug() << "\33[37m[AlephTestModule::_FaceAmrRefine]"
          << "\33[0m";
  Int32UniqueArray cells_lid;

  /*ENUMERATE_CELL(iCell, allCells()){
	 debug()<<"\t\t[FaceAmrRefine] cell #"<<(iCell).localId();
	 ENUMERATE_FACE(iFace, (*iCell).faces()){
		debug()<<"\t\t\t[FaceAmrRefine] face #"<<iFace->localId();
	 }
	 }*/

  ENUMERATE_CELL (iCell, MESH_ALL_ACTIVE_CELLS(mesh())) {
    if ((*iCell).level() == 1)
      continue;
    const Real T0 = m_cell_temperature[iCell];
    const Real T1 = values[m_cell_matrix_idx[iCell]];
    const Real ecart_relatif = ECART_RELATIF(T0, T1);
    Cell iItem = (*iCell);

    if (ecart_relatif > trigRefine) {
      //debug()<< "[AlephTestModule::_amrRefine] HIT ecart_relatif="<<ecart_relatif<<", cell_"<<(*iCell).localId();
      cells_lid.add((*iCell).localId());
      iItem.mutableItemBase().addFlags(ItemInternal::II_Refine);
    }
  }

  if (cells_lid.size() == 0)
    return false;

  debug() << "\33[37m[AlephTestModule::_amrRefine] NOW refine"
          << "\33[0m";
  MESH_MODIFIER_REFINE_ITEMS(mesh());
  mesh()->modifier()->endUpdate();

  // Now callBack the values
  CellInfoListView cells(mesh()->cellFamily());
  //ItemInternalList faces = mesh()->faceFamily()->itemsInternal();
  for (Integer i = 0, is = cells_lid.size(); i < is; ++i) {
    Int32 lid = cells_lid[i];
    Cell cell = cells[lid];

    for (Integer j = 0, js = CELL_NB_H_CHILDREN(cell); j < js; ++j) {
      //debug()<<"\t\t[amrRefineMesh] child cell #"<<cell.hChild(j).localId();
      m_cell_temperature[cells[CELL_H_CHILD(cell, j).localId()]] = m_cell_temperature[cells[lid]];
      auto faces = allCells().view()[CELL_H_CHILD(cell, j).localId()].toCell().faces();
      Integer index = 0;
      for (Face face : faces){
        if (face.isSubDomainBoundary()) {
          //debug() << "\t\t\t[amrRefineMesh] outer face #"<< iFace->localId()<<", index="<<iFace.index()<<", T="<<m_face_temperature[cell.face(iFace.index())];
          m_face_temperature[face] = m_face_temperature[cell.face(index)];
        }
        else {
          //debug() << "\t\t\t[amrRefineMesh] inner face #"<< iFace->localId();//<<", T="<<m_face_temperature[face.toFace()];
          m_face_temperature[face] = 0;
        }
        ++index;
      }
    }
  }

  debug() << "\33[37m[AlephTestModule::_amrRefine] done"
          << "\33[0m";
  return true;
}

/***********************************
 * AlephTestModule::_FaceAmrRefine *
 ***********************************/
bool AlephTestSchemeFaces::
amrCoarsen(RealArray& values, const Real trigCoarsen)
{
  ItacFunction(AlephTestSchemeFaces);
  if (!options()->amr)
    return false;
  debug() << "\33[37m[AlephTestModule::_FaceAmrCoarsen]"
          << "\33[0m";
  Int32UniqueArray parents_to_coarsen_lid;
  Int32UniqueArray children_to_coarsen_lid;
  mesh::FaceReorienter faceReorienter(mesh());
  mesh::DynamicMesh* dynMesh = dynamic_cast<mesh::DynamicMesh*>(mesh());
  CellInfoListView cells(mesh()->cellFamily());
  Int32UniqueArray faces_to_attach;
  Int32UniqueArray lids_to_be_attached;

  // Scan sur toutes les mailles, actives ET non actives
  ENUMERATE_CELL (iCell, allCells()) {
    Cell cell = *iCell;
    //onst Real Tp=m_cell_temperature[iCell];
    Real oldTs[4];
    Real newTs[4];
    Real sumTs = 0.0;
    bool coarsit = true;

    //debug()<<"true="<<true<<", false="<<false;
    if (cell.level() == 1)
      continue; // On évite de travailler sur les feuilles

    if (!CELL_HAS_H_CHILDREN(cell))
      continue; // On évite les mailles qui n'ont pas d'enfants

    // On scrute pour voir si tous les enfants sont bien actifs
    for (Integer j = 0, js = CELL_NB_H_CHILDREN(cell); j < js; ++j)
      coarsit &= CELL_H_CHILD(cell, j).isActive();
    if (coarsit == false)
      continue;

    //debug()<<"\t\t[FaceAmrCoarsen] parent cell Tp="<<Tp<<", "<<cell.localId();
    for (Integer j = 0, js = CELL_NB_H_CHILDREN(cell); j < js; ++j) {
      oldTs[j] = m_cell_temperature[cells[CELL_H_CHILD(cell, j).localId()]];
      newTs[j] = values[m_cell_matrix_idx[CELL_H_CHILD(cell, j)]];
      sumTs += newTs[j];
      coarsit &= (ECART_RELATIF(oldTs[j], newTs[j]) < trigCoarsen);
      //debug()<<"\t\t\t[FaceAmrCoarsen] child cell #"<<cell.hChild(j).localId()<<", oldT="<<oldTs[j]<<", newT="<<newTs[j];
    }
    if (coarsit == false)
      continue;

    // on regarde les écarts entre enfants
    for (Integer j = 0, js = CELL_NB_H_CHILDREN(cell); j < js; ++j)
      coarsit &= (ECART_RELATIF(newTs[j], newTs[(j + 1) % 4]) < trigCoarsen);
    if (coarsit == false)
      continue;

    debug() << "\n\t\t\t\t\t\t[FaceAmrCoarsen] parent cell to COARSE #" << cell.localId();

    // On rajoute l'lid du parent au pool de cells à coarsener
    parents_to_coarsen_lid.add((*iCell).localId());

    // Et on la tag comme Active
    (*iCell).mutableItemBase().removeFlags(ItemInternal::II_CoarsenInactive);
    ARCANE_ASSERT(((*iCell).isActive()), ("Parent not active!"));

    // On set la nouvell valeure de la maille
    m_cell_temperature[iCell] = sumTs / 4.0;

    // On rajoute les lids des enfants pour les remover par la suite
    for (Integer j = 0, js = CELL_NB_H_CHILDREN(cell); j < js; ++j) {
      children_to_coarsen_lid.add(CELL_H_CHILD(cell, j).localId());
      CELL_H_CHILD(cell, j).mutableItemBase().setFlags(CELL_H_CHILD(cell, j).itemBase().flags() | ItemInternal::II_Inactive); //II_Coarsen
    }

    // Now stare at the children to set up new faces' values
    Cell parent = cell;
    //const Cell &parent = cellFamily()->itemsInternal()[parents_to_coarsen_lid[i]];
    for( Face face : parent.faces()) {
      if ((!face.backCell().null()) && (!face.frontCell().null())) {
        debug() << "\t\t\t[FaceAmrCoarsen] FOCUS face #" << face.localId() << ", "
                << face.backCell().localId() << "->"
                << face.frontCell().localId();
        // Maintenant on va ratacher les voisins à notre parent
        const Cell& neighbour = (face.backCell().localId() == parent.localId()) ? face.frontCell() : face.backCell();
        debug() << "\t\t\t[FaceAmrCoarsen] neighbour #" << neighbour.localId() << ", level=" << neighbour.level();

        // Le voisin du parent doit être un parent
        ARCANE_ASSERT((neighbour.level() == 0), ("Wrong neighbour level!"));

        // Si le voisin est actif, la face en cours est celle qui doit être utilisée
        if (neighbour.isActive()) {
          debug() << "\t\t\t[FaceAmrCoarsen] neighbour is ACTIVE";
          debug() << "\t\t\t[FaceAmrCoarsen] hit: face_" << face.localId() << ": " << neighbour.localId() << "->" << parent.localId();
          //faces_to_attach.add(face.localId());
          //lids_to_be_attached.add(parent.localId());
          //dynMesh->trueFaceFamily().replaceBackCellToFace(face.internal(),neighbour.internal());//face.backCell().internal());
          //dynMesh->trueFaceFamily().replaceFrontCellToFace(face.internal(),parent.internal());//face.frontCell().internal());
          //faceReorienter.checkAndChangeOrientation(face.internal());
          continue;
        }

        {
          int iFound = 0;
          bool found[2];
          found[0] = found[1] = false;
          // Sinon, on racroche les faces des enfants de notre voisin vers notre parent
          for (Integer j = 0, js = CELL_NB_H_CHILDREN(neighbour); j < js; ++j) {
            //debug()<<"\t\t\t\t[FaceAmrCoarsen] neighbour child #"<<neighbour.hChild(j).localId();

            auto faces = CELL_H_CHILD(neighbour, j).faces();
            for (Face face : faces){
              if (face.backCell().null())
                continue;
              if (face.frontCell().null())
                continue;

              Cell other = (face.frontCell().localId() == CELL_H_CHILD(neighbour, j).localId()) ? face.backCell() : face.frontCell();
              //debug()<<"\t\t\t\t[FaceAmrCoarsen] neighbour child face #"<<face.localId()<<", "<< face.backCell().localId()<<"->"<< face.frontCell().localId();

              // Si l''autre' est un parent, c'est pas ce que l'on cherche
              if (other.level() == 0)
                continue;

              //debug()<<"\t\t\t\t[FaceAmrCoarsen] parent id="<<other.hParent().localId();
              if (CELL_H_PARENT(other).localId() != parent.localId())
                continue;

              debug() << "\t\t\t\t[FaceAmrCoarsen] hit: face_" << face.localId() << ": " << CELL_H_CHILD(neighbour, j).localId() << "->" << parent.localId();
              faces_to_attach.add(face.localId());
              lids_to_be_attached.add(parent.localId());
              found[iFound++] = true;
              break;
            }
            if (iFound == 2)
              break;
          }
        }
        //  ARCANE_ASSERT(((found[0]==true)&&(found[1]==true)),("Not found"));
      }
      else if (face.backCell().null()) {
        debug() << "\t\t\t[FaceAmrCoarsen] skip face #" << face.localId() << ", ?->" << face.frontCell().localId() << " ";
      }
      else {
        debug() << "\t\t\t[FaceAmrCoarsen] skip face #" << face.localId() << ", " << face.backCell().localId() << "->?"
                << " ";
      }
    }
  }

  // S'il n'y a rien à faire, on exit
  if (parents_to_coarsen_lid.size() == 0)
    return false;

  CellInfoListView cells_view(mesh()->cellFamily());
  FaceInfoListView faces_view(mesh()->faceFamily());

  // On remove les mailles
  for (Integer j = 0, js = children_to_coarsen_lid.size(); j < js; ++j) {
    //const Cell &cell=cellFamily()->itemsInternal()[children_to_coarsen_lid[j]];
    //debug()<<"\t\t[FaceAmrCoarsen] REMOVING CELL_"<<children_to_coarsen_lid[j];
    dynMesh->trueCellFamily().removeCell(cells_view[children_to_coarsen_lid[j]]);
    //dynMesh->trueCellFamily().detachCell(cellFamily()->itemsInternal()[children_to_coarsen_lid[j]]);
    /* ENUMERATE_FACE(iFace, cell.faces()){
		dynMesh->trueFaceFamily().populateBackFrontCellsFromParentFaces(cell.internal());

		}*/
  }
  //mesh()->modifier()->endUpdate();

  // On ré-attache les faces
  for (Integer j = 0, js = faces_to_attach.size(); j < js; ++j) {
    Face face = faces_view[faces_to_attach[j]];
    if (face.itemBase().flags() & ItemInternal::II_HasBackCell) {
      debug() << "\t\t[FaceAmrCoarsen] NOW patch face_" << faces_to_attach[j] << ": " << face.backCell().localId() << "->" << lids_to_be_attached[j];
      //dynMesh->trueFaceFamily().replaceFrontCellToFace(face.internal(),cellFamily()->itemsInternal()[lids_to_be_attached[j]]);
      dynMesh->trueFaceFamily().addFrontCellToFace(face, cells_view[lids_to_be_attached[j]]);
    }
    else {
      debug() << "\t\t[FaceAmrCoarsen] NOW patch face_" << faces_to_attach[j] << ": " << face.frontCell().localId() << "->" << lids_to_be_attached[j];
      //dynMesh->trueFaceFamily().replaceBackCellToFace(face.internal(),cellFamily()->itemsInternal()[lids_to_be_attached[j]]);
      dynMesh->trueFaceFamily().addBackCellToFace(face, cells_view[lids_to_be_attached[j]]);
    }
    faceReorienter.checkAndChangeOrientation(face);
  }
  //faceFamily()->endUpdate();

  // On met à jour nos changements
  mesh()->modifier()->endUpdate();

  debug() << "\t\t[FaceAmrCoarsen] nbCell=" << MESH_ALL_ACTIVE_CELLS(mesh()).size();

  //dynMesh->trueCellFamily().allItems().internal()->clear();//m_all_active_cell_group=0;
  /*ENUMERATE_CELL(iCell, allCells()){
	 debug()<<"\t\t[dump] cell  #"<<(iCell).localId()<<" ("<<(*iCell).isActive()<<")";
	 ENUMERATE_FACE(iFace, (*iCell).faces()){
		if ((!iFace->backCell().null())&&(!iFace->frontCell().null())){
		  debug()<<"\t\t\t[dump] face #"<<iFace->localId()<<", "\
				  <<iFace->backCell().localId()<<"->"\
				  <<iFace->frontCell().localId()\
				  <<" ("<<iFace->isActive()<<")";
		}else if (iFace->backCell().null()){
		  debug()<<"\t\t\t[dump] face #"<<iFace->localId()<<", ?->" \
				  <<iFace->frontCell().localId()\
				  <<" ("<<iFace->isActive()<<")";
		}else{
		  debug()<<"\t\t\t[dump] face #"<<iFace->localId()<<", " \
				  <<iFace->backCell().localId()<<"->?"							\
				  <<" ("<<iFace->isActive()<<")";
		}
	 }
	 }*/

  // Et on informe l'appellant comme quoi il y a eu des changements
  debug() << "[AlephTestModule::FaceAmrCoarsen] done";
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
