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
#include <arcane/core/SimpleSVGMeshExporter.h>

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
  void _buildCellByDirectionList();
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

namespace
{
/*!
 * A partir de la maille \a cell, parcours le maillage dans la direction
 * suivant la \a i-ème face jusqu'à arriver au bord et retourne cette maille
 * d'indice
 * Retourne la dernière maille
 */
Cell
_getLastCellFollowingDirection(Cell cell,Int32 i)
{
  Cell boundary_cell = cell;
  do {
    boundary_cell = cell;
    cell = cell.face(i).oppositeCell(cell);
  } while (!cell.null());
  return boundary_cell;
}

void
_addCellFollowingDirection(Array<Int32>& v,Cell cell,Int32 i)
{
  Cell current_cell = cell;
  do {
    current_cell = cell;
    v.add(current_cell.localId());
    cell = cell.face(i).oppositeCell(cell);
  } while (!cell.null());
}

}

void HoneyCombHeatModule::
_buildCellByDirectionList()
{
  // La direction 0 est vers le bas à droite.
  // La direction 1 est vers le haut à gauche
  // La direction 2 est vers le bas à gauche
  // Les directions 4, 5 et 6 sont construites en inversant la liste des mailles
  // respectivement des dimensions 0, 1 et 2.

  // Direction 0 -> face_index = 4
  // Direction 1 -> face_index = 1
  // Direction 2 -> face_index = 2

  // Valeurs des directions pour un maillage de 5 couches.

  // Indice Z= 0 [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  //              18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
  //              34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
  //              50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

  // Indice Z= 1 [ 4, 3, 2, 1, 0, 10, 9, 8, 7, 6, 5, 17, 16, 15, 14, 13, 12, 11,
  //              25, 24, 23, 22, 21, 20, 19, 18, 34, 33, 32, 31, 30, 29, 28, 27,
  //              26, 42, 41, 40, 39, 38, 37, 36, 35, 49, 48, 47, 46, 45, 44, 43,
  //              55, 54, 53, 52, 51, 50, 60, 59, 58, 57, 56]

  // Indice Z= 2 [ 34, 25, 17, 10, 4, 42, 33, 24, 16, 9, 3, 49, 41, 32, 23, 15, 8,
  //                2, 55, 48, 40, 31, 22, 14, 7, 1, 60, 54, 47, 39, 30, 21, 13, 6,
  //                0, 59, 53, 46, 38, 29, 20, 12, 5, 58, 52, 45, 37, 28, 19, 11, 57,
  //               51, 44, 36, 27, 18, 56, 50, 43, 35, 26]

  // Indice Z= 3 [ 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44,
  //               43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27,
  //               26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
  //                9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

  // Indice Z= 4 [ 56, 57, 58, 59, 60, 50, 51, 52, 53, 54, 55, 43, 44, 45, 46, 47, 48,
  //               49, 35, 36, 37, 38, 39, 40, 41, 42, 26, 27, 28, 29, 30, 31, 32, 33,
  //               34, 18, 19, 20, 21, 22, 23, 24, 25, 11, 12, 13, 14, 15, 16, 17, 5,
  //                6, 7, 8, 9, 10, 0, 1, 2, 3, 4]

  // Indice Z= 5 [ 26, 35, 43, 50, 56, 18, 27, 36, 44, 51, 57, 11, 19, 28, 37, 45, 52,
  //               58, 5, 12, 20, 29, 38, 46, 53, 59, 0, 6, 13, 21, 30, 39, 47, 54, 60,
  //                1, 7, 14, 22, 31, 40, 48, 55, 2, 8, 15, 23, 32, 41, 49, 3, 9, 16,
  //               24, 33, 42, 4, 10, 17, 25, 34]

  // Indice local de la face par directions pour Arcane avec le HoneyCombMeshGenerator:
  /*
   *     3
   *    / \
   *   4   2
   *   |   |
   *   5   1
   *    \ /
   *     0
   */

  static constexpr int NB_DIRECTION = 6;

  UniqueArray<UniqueArray<Int32>> boundary_cells(NB_DIRECTION);
  // Calcule les mailles au bord pour chaque direction.
  // Pour la direction 0, on commence par la maille 0 (qui est à gauche en bas)

  // Le code suivant ne fonctionne qu'en 2D et en séquentiel
  IItemFamily* cell_family = defaultMesh()->cellFamily();

  std::array<Cell, NB_DIRECTION> begin_cells;
  begin_cells[0] = cell_family->findOneItem(0);
  info() << "CELL0=" << ItemPrinter(begin_cells[0]);

  // Cherche les 6 mailles de bord.
  {
    // Pour trouver la première maille de la direction 1 on part de la maille 0
    // et on va en bas à droite jusqu'au bord
    // puis vers le haut à droite (2 ème direction)
    // puis vers le haut (3 ème direction)
    // puis vers le haut à gauche (4 ème direction)
    // puis vers le bas à gauche (5 ème direction)
    std::array<Int32, NB_DIRECTION> next_directions = { 0, 1, 2, 3, 4, 5 };

    for (int i = 1; i < NB_DIRECTION; ++i) {
      Cell cell = _getLastCellFollowingDirection(begin_cells[i - 1], next_directions[i]);
      begin_cells[i] = cell;
      info() << "CELL[" << i << "]=" << ItemPrinter(cell);
    }
  }

  // Indice local de la face dans la direction souhaité
  std::array<Int32, NB_DIRECTION> search_directions = { 1, 4, 5, 4, 1, 2 };

  // Pour chaque direction, on part de deux mailles de bord et on prend
  // toutes les mailles dans la direction 'direction_for_next'.
  // On obtient une liste A de mailles. Pour avoir la liste des toutes les
  // mailles dans une direction on prend chaque maille de A et on ajoute toutes
  // les mailles qui partent de A
  struct BeginAndDirection
  {
    int begin_cell;
    int direction_for_next;
  };

  std::array<std::array<BeginAndDirection, 2>, NB_DIRECTION> begin_and_directions = { {
  { BeginAndDirection{ 0, 3 }, BeginAndDirection{ 5, 2 } }, // direction 0
  { BeginAndDirection{ 1, 2 }, BeginAndDirection{ 2, 3 } }, // direction 1
  { BeginAndDirection{ 2, 3 }, BeginAndDirection{ 3, 4 } }, // direction 2
  { BeginAndDirection{ 3, 0 }, BeginAndDirection{ 2, 5 } }, // direction 3
  { BeginAndDirection{ 4, 5 }, BeginAndDirection{ 5, 0 } }, // direction 4
  { BeginAndDirection{ 5, 0 }, BeginAndDirection{ 0, 1 } } // direction 5
  } };

  for (int i = 0; i < NB_DIRECTION; ++i) {
    const Int32 search_dir = search_directions[i];
    for (int j = 0; j < 2; ++j) {
      BeginAndDirection x = begin_and_directions[i][j];
      Cell cell = begin_cells[x.begin_cell];
      Cell next_cell;
      if (j==0){
        next_cell = begin_cells[begin_and_directions[i][j+1].begin_cell];
      }
      const Int32 next = x.direction_for_next;
      do {
        _addCellFollowingDirection(boundary_cells[i], cell, search_dir);
        cell = cell.face(next).oppositeCell(cell);
        if (next_cell==cell)
          break;
      } while (!cell.null());
    }
  }

  for (int i = 0; i < NB_DIRECTION; ++i) {
    info() << "DIRECTIONS: I=" << i << " list=" << boundary_cells[i] << "\n";
  }
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

  if (defaultMesh()->dimension() == 2) {
    std::ofstream ofile("honeycomb.svg");
    SimpleSVGMeshExporter exporter(ofile);
    exporter.write(allCells());
  }

  _buildCellByDirectionList();

  // Calcule le centre des mailles
  VariableNodeReal3& nodes_coords(mesh()->nodesCoordinates());
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Int32 nb_node = cell.nbNode();
    Real3 center;
    for (NodeLocalId node : cell.nodeIds()) {
      center += nodes_coords[node];
    }
    center /= nb_node;
    m_cell_center[cell] = center;
    if (is_verbose)
      info() << "Cell " << ItemPrinter(cell) << " center=" << center;
  }

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
