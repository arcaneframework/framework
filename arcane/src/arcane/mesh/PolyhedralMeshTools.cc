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

#include "PolyhedralMeshTools.h"
#include "PolyhedralMesh.h"
#include "arcane/ArcaneTypes.h"

#include <neo/Mesh.h>

#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkNew.h>
#include <vtkCellIterator.h>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace mesh
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  namespace PolyhedralMeshTools
  {

    void VtkReader::
    read(const String& filename, PolyhedralMesh& mesh1){
      vtkNew<vtkUnstructuredGridReader> vtk_grid_reader;
      vtk_grid_reader->SetFileName(filename.localstr());
      vtk_grid_reader->Update();
      auto vtk_grid = vtk_grid_reader->GetOutput();
      std::cout << "-- VTK GRID READ " << " NB CELLS  " << vtk_grid->GetNumberOfCells() << std::endl;
      // Parse cells
      auto cell_iter = vtk_grid->vtkDataSet::NewCellIterator();
      cell_iter->InitTraversal();
      vtkIdType* cell_faces {nullptr};
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

  } // namespace PolyhedralMeshTools

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane