// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMeshTools                             (C) 2000-2022             */
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

  class VtkReader{

   public:
    VtkReader(const String&  filename);

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

   private:
    const String& m_filename;
    vtkNew<vtkUnstructuredGridReader> m_vtk_grid_reader;
    UniqueArray<Int64> m_cell_uids, m_node_uids, m_face_uids, m_edge_uids;
    UniqueArray<Int64> m_face_node_uids, m_edge_node_uids, m_cell_node_uids;
    UniqueArray<Int64> m_face_cell_uids, m_edge_cell_uids;
    UniqueArray<Int32> m_face_nb_nodes, m_cell_nb_nodes, m_face_nb_cells, m_edge_nb_cells;

    std::pair<bool, Int32> _findFace(Int64ConstArrayView face_nodes, Int64ConstArrayView face_node_uids, Int32ConstArrayView face_nb_nodes);
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
