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

    Arcane::Int64ConstArrayView cellNodes();

    Int32ConstArrayView cellNbNodes();

   private:
    const String& m_filename;
    vtkNew<vtkUnstructuredGridReader> m_vtk_grid_reader;
    UniqueArray<Int64> m_cell_uids, m_node_uids, m_face_uids, m_edge_uids;
    UniqueArray<Int64> m_face_node_uids, m_edge_node_uids, m_cell_node_uids;
    UniqueArray<Int32> m_face_nb_nodes, m_cell_nb_nodes;

    bool _findFace(UniqueArray<Int64> face_nodes);
    bool _findEdge(UniqueArray<Int64> edge_nodes);
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
