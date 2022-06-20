// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMeshTools.cc                          (C) 2000-2022             */
/*                                                                           */
/* Tools for polyhedral mesh                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/MeshUtils.h"
#include "arcane/mesh/PolyhedralMesh.h"
#include "arcane/mesh/PolyhedralMeshTools.h"
#include "arcane/utils/ArcaneGlobal.h"

#include <neo/Mesh.h>

#include <vtkCellIterator.h>
#include <vtkIdTypeArray.h>

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::mesh::PolyhedralMeshTools::VtkReader::
VtkReader(const String& filename) : m_filename{filename}
{
  if (filename.empty())
    ARCANE_FATAL("filename for polyhedral vtk mesh is empty.");
  m_vtk_grid_reader->SetFileName(filename.localstr());
  m_vtk_grid_reader->Update();
  auto* vtk_grid = m_vtk_grid_reader->GetOutput();
  if (!vtk_grid)
    ARCANE_FATAL("Cannot read vtk polyhedral file {0}",filename);

  std::cout << "-- VTK GRID READ "
            << " NB CELLS  " << vtk_grid->GetNumberOfCells() << std::endl;
  // Parse cells
  auto* cell_iter = vtk_grid->vtkDataSet::NewCellIterator();
  cell_iter->InitTraversal();
  vtkIdType* cell_faces{ nullptr };
  vtkIdType nb_faces = 0;
  while (!cell_iter->IsDoneWithTraversal()) {
    std::cout << "---- visiting cell id " << cell_iter->GetCellId() << std::endl;
    std::cout << "----   cell number of faces " << cell_iter->GetNumberOfFaces() << std::endl;
    std::cout << "----   cell number of points " << cell_iter->GetNumberOfPoints() << std::endl;
    vtk_grid->GetFaceStream(cell_iter->GetCellId(), nb_faces, cell_faces);
    for (auto iface = 0; iface < nb_faces; ++iface) {
      auto face_nb_nodes = *cell_faces++;
      std::cout << "----      has face with " << face_nb_nodes << " nodes. Node ids : ";
      for (int inode = 0; inode < face_nb_nodes; ++inode) {
        std::cout << *cell_faces++;
      }
      std::cout << std::endl;
    }
    cell_iter->GoToNextCell();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::Int64ConstArrayView Arcane::mesh::PolyhedralMeshTools::VtkReader::
cellUids()
{
  if (m_cell_uids.empty()) {
    auto* vtk_grid = m_vtk_grid_reader->GetOutput();
    m_cell_uids.reserve(vtk_grid->GetNumberOfCells());
    auto* cell_iter = vtk_grid->NewCellIterator();
    cell_iter->InitTraversal();
    while (!cell_iter->IsDoneWithTraversal()) {
      m_cell_uids.push_back(cell_iter->GetCellId());
      cell_iter->GoToNextCell();
    }
  }
  return m_cell_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::Int64ConstArrayView Arcane::mesh::PolyhedralMeshTools::VtkReader::
nodeUids()
{
  if (m_node_uids.empty()) {
    auto* vtk_grid = m_vtk_grid_reader->GetOutput();
    auto nb_nodes = vtk_grid->GetNumberOfPoints();
    m_node_uids.reserve(nb_nodes);
    for (int node_index = 0; node_index < nb_nodes; ++node_index) {
      m_node_uids.push_back(node_index);
    }
  }
  return m_node_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::Int64ConstArrayView Arcane::mesh::PolyhedralMeshTools::VtkReader::faceUids()
{
  if (m_face_uids.empty()) {
    auto* vtk_grid = m_vtk_grid_reader->GetOutput();
    auto* cell_iter = vtk_grid->NewCellIterator();
    cell_iter->InitTraversal();
    vtkIdType nb_face_estimation = 0;
    while (!cell_iter->IsDoneWithTraversal()) {
      vtkIdType cell_nb_faces = 0;
      vtkIdType* points{ nullptr };
      vtk_grid->GetFaceStream(cell_iter->GetCellId(), cell_nb_faces, points);
      nb_face_estimation += cell_nb_faces;
      cell_iter->GoToNextCell();
    }
    m_face_uids.reserve(nb_face_estimation);
    auto* faces = vtk_grid->GetFaces();
    // This array contains the face info per cells (cf. vtk file)
    // first_cell_nb_faces first_cell_first_face_nb_nodes first_cell_first_face_node_1 ... first_cell_first_face_node_n first_cell_second_face_nb_nodes etc

    if (!faces) {
      ARCANE_FATAL("Mesh {0} is not polyhedral: faces are not defined",m_filename);
    }
    Int64 face_uid = 0;
    auto face_info_size = faces->GetNumberOfValues();
    m_face_node_uids.reserve(face_info_size);
    m_face_nb_nodes.reserve(nb_face_estimation);
    UniqueArray<Int64> current_face_nodes, sorted_current_face_nodes;
    current_face_nodes.reserve(10);
    sorted_current_face_nodes.reserve(10);
    for (int face_info_index = 0; face_info_index < face_info_size; ) {
      auto current_cell_nb_faces = Int32 (faces->GetValue(face_info_index++));
      for (auto face_index = 0; face_index < current_cell_nb_faces; ++face_index) {
        auto current_face_nb_nodes = Int32 (faces->GetValue(face_info_index++));
        for (int node_index = 0; node_index < current_face_nb_nodes; ++node_index) {
          current_face_nodes.push_back(faces->GetValue(face_info_index++));
        }
        sorted_current_face_nodes.resize(current_face_nodes.size());
        mesh_utils::reorderNodesOfFace(current_face_nodes, sorted_current_face_nodes);
        if (!_findFace(sorted_current_face_nodes)) {
          m_face_uids.push_back(face_uid++); // todo parallel
          m_face_nb_nodes.push_back(current_face_nb_nodes);
          m_face_node_uids.addRange(sorted_current_face_nodes);
        }
        current_face_nodes.clear();
        sorted_current_face_nodes.clear();
      }
    }
  }
  std::cout << "================FACE NODES ==============" << std::endl;
  std::copy(m_face_node_uids.begin(), m_face_node_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
  std::cout << std::endl;
  std::copy(m_face_nb_nodes.begin(), m_face_nb_nodes.end(), std::ostream_iterator<Int64>(std::cout, " "));
  std::cout << std::endl;
  return m_face_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::Int64ConstArrayView Arcane::mesh::PolyhedralMeshTools::VtkReader::
edgeUids()
{
  // TODO check !!
  if (m_edge_uids.empty()) {
    auto vtk_grid = m_vtk_grid_reader->GetOutput();
    m_edge_uids.reserve(2*vtk_grid->GetNumberOfPoints());
    auto* faces = vtk_grid->GetFaces();
    // This array contains the face info per cells (cf. vtk file)
    // first_cell_nb_faces first_cell_first_face_nb_nodes first_cell_first_face_node_1 ... first_cell_first_face_node_n first_cell_second_face_nb_nodes etc

    if (!faces) {
      ARCANE_FATAL("Mesh {0} is not polyhedral: faces are not defined",m_filename);
    }
    Int64 edge_uid = 0;
    m_edge_node_uids.reserve(2*m_edge_uids.size());
    auto face_info_size = faces->GetNumberOfValues();
    for (int face_info_index = 0; face_info_index < face_info_size; ) {
      auto current_cell_nb_faces = Int32 (faces->GetValue(face_info_index++));
      for (auto face_index = 0; face_index < current_cell_nb_faces; ++face_index) {
        auto current_face_nb_nodes = Int32(faces->GetValue(face_info_index++));
        auto first_face_node_uid = Int32(faces->GetValue(face_info_index));
        UniqueArray<Int64> current_edge(2), sorted_edge(2);
        for (int node_index = 0; node_index < current_face_nb_nodes - 1; ++node_index) {
          current_edge = UniqueArray<Int64>{ faces->GetValue(face_info_index++), faces->GetValue(face_info_index) };
          mesh_utils::reorderNodesOfFace(current_edge, sorted_edge); // works for edges
          if (!_findEdge(sorted_edge)) {
            m_edge_uids.push_back(edge_uid++); // todo parallel
            m_edge_node_uids.addRange(sorted_edge);
          }
        }
        current_edge = UniqueArray<Int64>{ faces->GetValue(face_info_index++), first_face_node_uid };
        mesh_utils::reorderNodesOfFace(current_edge, sorted_edge); // works for edges
        if (!_findEdge(sorted_edge)) {
          m_edge_uids.push_back(edge_uid++); // todo parallel
          m_edge_node_uids.addRange(sorted_edge);
        }
      }
    }
  }
  std::cout << "================EDGE NODES ==============" << std::endl;
  std::copy(m_edge_node_uids.begin(), m_edge_node_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
  std::cout << std::endl;
  return m_edge_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Arcane::mesh::PolyhedralMeshTools::VtkReader::
_findFace(UniqueArray<Int64> face_nodes)
{
  // todo coder l'algo recherche : d'abord on vérifie nombre de noeuds puis on teste tant que l'id est égal (est-ce beaucoup plus rapide ?)
  auto it = std::search(m_face_node_uids.begin(), m_face_node_uids.end(), std::boyer_moore_searcher(face_nodes.begin(), face_nodes.end()));
  return it != m_face_node_uids.end();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Arcane::mesh::PolyhedralMeshTools::VtkReader::
_findEdge(UniqueArray<Int64> edge_nodes)
{
  auto it = std::search(m_edge_node_uids.begin(), m_edge_node_uids.end(), std::boyer_moore_searcher(edge_nodes.begin(), edge_nodes.end()));
  return it != m_edge_node_uids.end();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::Integer Arcane::mesh::PolyhedralMeshTools::VtkReader::
nbNodes()
{
  if (m_node_uids.empty()) nodeUids();
  return m_node_uids.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/