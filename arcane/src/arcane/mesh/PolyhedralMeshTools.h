// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMeshTools.h                                       (C) 2000-2022 */
/*                                                                           */
/* Tools for polyhedral mesh                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_POLYHEDRALMESHTOOLS_H
#define ARCANE_POLYHEDRALMESHTOOLS_H

#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkNew.h>

#include "arcane/ArcaneTypes.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace mesh
{

  class PolyhedralMesh;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  namespace PolyhedralMeshTools
  {

    class VtkReader
    {

     public:

      VtkReader(const String& filename);

      Int64ConstArrayView cellUids();
      Int64ConstArrayView nodeUids();
      Int64ConstArrayView faceUids();
      Int64ConstArrayView edgeUids();

      Integer nbNodes();

      Int64ConstArrayView cellNodes();
      Int32ConstArrayView cellNbNodes();

      Int64ConstArrayView faceNodes();
      Int32ConstArrayView faceNbNodes();

      Int64ConstArrayView edgeNodes();

      Int64ConstArrayView faceCells();
      Int32ConstArrayView faceNbCells();

      Int32ConstArrayView edgeNbCells();
      Int64ConstArrayView edgeCells();

      Int32ConstArrayView cellNbFaces();
      Int64ConstArrayView cellFaces();

      Int32ConstArrayView edgeNbFaces();
      Int64ConstArrayView edgeFaces();

      Int32ConstArrayView cellNbEdges();
      Int64ConstArrayView cellEdges();

      Int32ConstArrayView faceNbEdges();
      Int64ConstArrayView faceEdges();

      Int32ConstArrayView nodeNbCells();
      Int64ConstArrayView nodeCells();

      Int32ConstArrayView nodeNbFaces();
      Int64ConstArrayView nodeFaces();

      Int32ConstArrayView nodeNbEdges();
      Int64ConstArrayView nodeEdges();

      Real3ArrayView nodeCoords();

     private:

      const String& m_filename;
      vtkNew<vtkUnstructuredGridReader> m_vtk_grid_reader;
      Int64UniqueArray m_cell_uids, m_node_uids, m_face_uids, m_edge_uids;
      Int64UniqueArray m_face_node_uids, m_edge_node_uids, m_cell_node_uids;
      Int64UniqueArray m_face_cell_uids, m_edge_cell_uids, m_edge_face_uids;
      Int64UniqueArray m_cell_face_uids, m_cell_edge_uids, m_face_edge_uids;
      Int64UniqueArray m_node_cell_uids, m_node_face_uids, m_node_edge_uids;
      Int32UniqueArray m_face_nb_nodes, m_cell_nb_nodes, m_face_nb_cells;
      Int32UniqueArray m_edge_nb_cells, m_edge_nb_faces, m_cell_nb_faces;
      Int32UniqueArray m_node_nb_cells, m_node_nb_faces, m_node_nb_edges;
      Int32UniqueArray m_cell_nb_edges, m_face_nb_edges, m_face_uid_indexes;
      Int32UniqueArray m_cell_face_indexes;
      Real3UniqueArray m_node_coordinates;

      std::pair<bool, Int32> _findFace(Int64ConstArrayView face_nodes, Int64ConstArrayView face_node_uids, Int32ConstArrayView face_nb_nodes);
      template <typename Connectivity2DArray>
      void _flattenConnectivity(Connectivity2DArray connected_item_2darray, Int32Span nb_connected_item_per_source_item, Int64UniqueArray& connected_item_array);
    };

  } // namespace PolyhedralMeshTools

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_POLYHEDRALMESHTOOLS_H
