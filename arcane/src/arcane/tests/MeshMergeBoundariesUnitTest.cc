// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMergeBoundariesUnitTest.cc                              (C) 2000-2024 */
/*                                                                           */
/* Service de test de la fusion de la frontière de plusieurs maillages.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/IMeshUtilities.h"

#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la fusion de la frontière de plusieurs maillages.
 *
 * Ce service permet de prendre N maillages de même dimension et
 * de générer un nouveau maillage contenant uniquement la frontière ce ces
 * N maillages.
 *
 * /note Pour l'instant cela ne fonctionne que en séquentiel.
 */
class MeshMergeBoundariesUnitTest
: public BasicUnitTest
{
 public:

  explicit MeshMergeBoundariesUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MeshMergeBoundariesUnitTest,
                        ServiceProperty("MeshMergeBoundariesUnitTest", ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IUnitTest));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMergeBoundariesUnitTest::
MeshMergeBoundariesUnitTest(const ServiceBuildInfo& sbi)
: BasicUnitTest(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeBoundariesUnitTest::
executeTest()
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm = mesh()->parallelMng();
  IMeshMng* mesh_mng = mesh()->meshMng();

  Int32 mesh_dim = mesh()->dimension();
  if (mesh_dim != 2)
    ARCANE_FATAL("Bad value '{0}' for mesh dimension. Valid values is '2'", mesh_dim);

  for (IMesh* mesh : sd->meshes()) {
    info() << "MESH name=" << mesh->name();
    Int32 md = mesh->dimension();
    if (md != mesh_dim)
      ARCANE_FATAL("Bad dimension '{0}' for mesh '{1}'. The mesh should have dimension {2}",
                   md, mesh->name(), mesh_dim);
  }

  MeshBuildInfo mbi("BoundaryMesh");
  mbi.addParallelMng(makeRef(pm));

  IPrimaryMesh* boundary_mesh = mesh_mng->meshFactoryMng()->createMesh(mbi);
  boundary_mesh->setDimension(mesh_dim);
  UniqueArray<Int64> cells_infos;
  cells_infos.reserve(10000);

  // Contient l'offset à ajouter aux uniqueId()
  // pour éviter qu'ils soient identiques
  Int64 offset_cell_uid = 0;
  Int64 offset_node_uid = 0;
  Int32 nb_cell = 0;
  std::unordered_map<Int64, Real3> node_to_coord_map;
  for (IMesh* mesh : sd->meshes()) {
    VariableNodeReal3& node_coord = mesh->nodesCoordinates();
    FaceGroup outer_faces = mesh->allCells().outerFaceGroup();
    Int64 max_cell_uid = 0;
    Int64 max_node_uid = 0;
    info() << "Add faces from mesh=" << mesh->name() << " nb_face=" << outer_faces.size();
    ENUMERATE_ (Face, iface, outer_faces) {
      Face face = *iface;
      Cell cell = face.cell(0);
      if (!cell.isOwn())
        continue;
      Int64 uid = cell.uniqueId();
      Int16 cell_type = cell.type();
      cells_infos.add(cell_type);
      Int64 cell_uid = uid + offset_cell_uid;
      cells_infos.add(cell_uid);
      max_cell_uid = std::max(max_cell_uid, cell_uid);
      for (Node node : cell.nodes()) {
        Int64 node_uid = node.uniqueId().asInt64() + offset_node_uid;
        max_node_uid = std::max(max_node_uid, node_uid);
        node_to_coord_map[node_uid] = node_coord[node];
        cells_infos.add(node_uid);
      }
      ++nb_cell;
    }
    offset_cell_uid = 1 + mesh->parallelMng()->reduce(Parallel::ReduceMax, max_cell_uid);
    offset_node_uid = 1 + mesh->parallelMng()->reduce(Parallel::ReduceMax, max_node_uid);
  }
  boundary_mesh->allocateCells(nb_cell, cells_infos);

  // Maintenant remplit les coordonnées des noeuds
  {
    VariableNodeReal3& node_coord = boundary_mesh->nodesCoordinates();
    ENUMERATE_ (Node, inode, boundary_mesh->allNodes()) {
      Node node = *inode;
      node_coord[node] = node_to_coord_map[node.uniqueId()];
    }
  }

  // Ecrit le maillage
  boundary_mesh->utilities()->writeToFile("boundary.vtk", "VtkLegacyMeshWriter");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
