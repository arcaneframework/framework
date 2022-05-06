// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HoneycombHeatModule.cc                                      (C) 2000-2022 */
/*                                                                           */
/* Module HoneycombHeatModule of honeycomb_heat sample.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "HoneyCombHeat_axl.h"
#include <arcane/ITimeLoopMng.h>
#include <arcane/IItemFamily.h>
#include <arcane/IndexedItemConnectivityView.h>
#include <arcane/IMesh.h>
#include <arcane/UnstructuredMeshConnectivity.h>
#include <arcane/ItemPrinter.h>
#include <arcane/mesh/IncrementalItemConnectivity.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module HoneyCombHeatModule.
 */
class HoneyCombHeatModule
: public ArcaneHoneyCombHeatObject
{
 public:

  explicit HoneyCombHeatModule(const ModuleBuildInfo& mbi);

 public:

  /*!
   * \brief Méthode appelée à chaque itération.
   */
  void compute() override;
  /*!
   * \brief Méthode appelée lors de l'initialisation.
   */
  void startInit() override;

  /** Retourne le numéro de version du module */
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 private:

  //! Connectivités standards du maillage
  UnstructuredMeshConnectivityView m_mesh_connectivity_view;

  //! Vue sur la connectivité Maille<->Maille par les faces
  IndexedCellCellConnectivityView m_cell_cell_connectivity_view;

  /*!
   * \brief Index de la face (donc entre 0 et 5 (2D) ou 7 (3D)) dans la maille
   * voisine pour la i-ème maille connectée.
   *
   * La valeur de \a i correspond au parcours via m_cell_cell_connectivity_view.
   * Pour une maille interne, \a i est aussi l'index de la face dans cette propre maille
   * mais ce n'est pas le cas pour une maille externe (c'est à dire une maille qui a
   * au moins une face non connectée à une autre maille).
   */
  VariableCellArrayInt32 m_cell_neighbour_face_index;

  /*!
   * \brief Index de la face (donc entre 0 et 5 (2D) ou 7 (3D)) dans notre maille
   * pour la i-ème maille connectée.
   *
   * La valeur de \a i correspond au parcours via m_cell_cell_connectivity_view.
   * Pour une maille interne, \a i est aussi l'index de la face dans cette propre maille
   * mais ce n'est pas le cas pour une maille externe (c'est à dire une maille qui a
   * au moins une face non connectée à une autre maille).
   */
  VariableCellArrayInt32 m_cell_current_face_index;

 private:

  void _applyBoundaryCondition();

  Int32 _getNeighbourFaceIndex(CellLocalId cell, CellLocalId neighbour_cell);
  Int32 _getCurrentFaceIndex(CellLocalId cell, FaceLocalId face);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HoneyCombHeatModule::
HoneyCombHeatModule(const ModuleBuildInfo& mbi)
: ArcaneHoneyCombHeatObject(mbi)
, m_cell_neighbour_face_index(VariableBuildInfo(mbi.meshHandle(), "CellNeighbourFaceIndex"))
, m_cell_current_face_index(VariableBuildInfo(mbi.meshHandle(), "CellCurrentFaceIndex"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyCombHeatModule::
compute()
{
  info() << "Module HoneyCombHeatModule COMPUTE";

  // Stop code after 10 iterations
  if (m_global_iteration() > 500) {
    subDomain()->timeLoopMng()->stopComputeLoop(true);
    return;
  }

  // Mise a jour de la temperature aux noeuds en prenant la moyenne
  // valeurs aux mailles voisines
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;
    Real sumt = 0;
    for (Cell cell : node.cells())
      sumt += m_cell_temperature[cell];
    m_node_temperature[inode] = sumt / node.nbCell();
  }
  m_node_temperature.synchronize();

  // Mise a jour de la temperature aux mailles en prenant la moyenne
  // des valeurs aux noeuds voisins
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    Real sumt = 0;
    for (Node node : cell.nodes())
      sumt += m_node_temperature[node];
    m_cell_temperature[icell] = sumt / cell.nbNode();
  }

  _applyBoundaryCondition();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne l'index dans la liste des faces de 'neighbour_cell' de
 * la face commune entre 'cell' et 'neighbour_cell'.
 */
Int32 HoneyCombHeatModule::
_getNeighbourFaceIndex(CellLocalId cell, CellLocalId neighbour_cell)
{
  auto cell_face_cv = m_mesh_connectivity_view.cellFace();

  Int32 face_neighbour_index = 0;
  // Recherche la face commune entre 'neighbour_cell' et 'cell'.
  for (FaceLocalId neighbour_face : cell_face_cv.faces(neighbour_cell)) {
    for (FaceLocalId current_face : cell_face_cv.faces(cell)) {
      if (current_face == neighbour_face)
        return face_neighbour_index;
    }
    ++face_neighbour_index;
  }
  ARCANE_FATAL("No common face between the two cells '{0}' and '{1}'", cell, neighbour_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyCombHeatModule::
startInit()
{
  info() << "Module HoneyCombHeatModule INIT";
  m_cell_temperature.fill(0.0);
  m_node_temperature.fill(0.0);

  // Initialise le pas de temps à une valeur fixe
  m_global_deltat = 1.0;

  const bool is_verbose = false;

  m_mesh_connectivity_view.setMesh(mesh());

  // Créé une connectivité Maille/Maille sur les mailles voisines
  IItemFamily* cell_family = mesh()->cellFamily();
  CellGroup cells = cell_family->allItems();
  // NOTE: l'objet est automatiquement détruit par le maillage
  auto* cn = new mesh::IncrementalItemConnectivity(cell_family, cell_family, "NeighbourCellCell");
  ENUMERATE_CELL (icell, cells) {
    Cell cell = *icell;
    Integer nb_face = cell.nbFace();
    cn->notifySourceItemAdded(cell);
    for (Integer i = 0; i < nb_face; ++i) {
      Face face = cell.face(i);
      if (face.nbCell() == 2) {
        Cell opposite_cell = (face.backCell() == cell) ? face.frontCell() : face.backCell();
        cn->addConnectedItem(cell, opposite_cell);
      }
    }
  }
  m_cell_cell_connectivity_view = cn->connectivityView();

  const Int32 max_neighbour = (mesh()->dimension() == 3) ? 8 : 6;
  m_cell_neighbour_face_index.resize(max_neighbour);
  m_cell_neighbour_face_index.fill(NULL_ITEM_LOCAL_ID);
  m_cell_current_face_index.resize(max_neighbour);
  m_cell_current_face_index.fill(NULL_ITEM_LOCAL_ID);

  // Calcul l'index de la face voisine pour chaque maille connectée
  ENUMERATE_ (Cell, icell, cells) {
    Cell cell = *icell;
    Int32 local_cell_index = 0;
    for (CellLocalId neighbour_cell : m_cell_cell_connectivity_view.cells(icell)) {

      Int32 neighbour_face_index = _getNeighbourFaceIndex(cell, neighbour_cell);
      Int32 current_face_index = _getNeighbourFaceIndex(neighbour_cell, cell);

      if (is_verbose)
        info() << "Cell=" << cell.uniqueId() << " I=" << local_cell_index
               << " neighbour_cell_local_id=" << neighbour_cell
               << " face_index_in_neighbour_cell=" << neighbour_face_index
               << " face_index_in_current_cell=" << current_face_index;

      m_cell_neighbour_face_index[icell][local_cell_index] = neighbour_face_index;
      m_cell_current_face_index[icell][local_cell_index] = current_face_index;
      ++local_cell_index;
    }
  }

  // Vérifie que tout est OK
  {
    auto cell_face_cv = m_mesh_connectivity_view.cellFace();

    ENUMERATE_ (Cell, icell, cells) {
      Cell cell = *icell;
      auto neighbour_cells_id = m_cell_cell_connectivity_view.cells(icell);
      for (Int32 i = 0; i < neighbour_cells_id.size(); ++i) {
        CellLocalId neighbour_cell_id = neighbour_cells_id[i];

        Int32 neighbour_face_index = m_cell_neighbour_face_index[icell][i];
        Int32 current_face_index = m_cell_current_face_index[icell][i];

        if (is_verbose)
          info() << "Check: Cell=" << cell.uniqueId() << " I=" << i
                 << " neighbour_cell_local_id=" << neighbour_cell_id
                 << " face_index_in_neighbour_cell=" << neighbour_face_index
                 << " face_index_in_current_cell=" << current_face_index;

        FaceLocalId neighbour_face_id = cell_face_cv.faces(neighbour_cell_id)[neighbour_face_index];
        FaceLocalId current_face_id = cell_face_cv.faces(cell)[current_face_index];
        if (neighbour_face_id != current_face_id)
          ARCANE_FATAL("Bad face neighbour={0} current={1} cell={2}", neighbour_face_id, current_face_id, ItemPrinter(cell));
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyCombHeatModule::
_applyBoundaryCondition()
{
  // Les 10 premières mailles ont une température fixe.
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    if (cell.uniqueId() < 10)
      m_cell_temperature[cell] = 10000.0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_HONEYCOMBHEAT(HoneyCombHeatModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
