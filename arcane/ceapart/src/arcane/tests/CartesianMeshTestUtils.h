// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshTestUtils.cc                                   (C) 2000-2021 */
/*                                                                           */
/* Fonctions utilitaires pour les tests de 'CartesianMesh'.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CEA_TESTS_CARTESIANMESHTESTUTILS_H
#define ARCANE_CEA_TESTS_CARTESIANMESHTESTUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ICartesianMesh;
}

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe utilitaire pour tester les variantes de 'CartesianMesh'
 */
class CartesianMeshTestUtils
: public TraceAccessor
{
 public:

  explicit CartesianMeshTestUtils(ICartesianMesh* cm);
  ~CartesianMeshTestUtils();

 public:

  void testAll();

 public:

  void checkSameId(Face item,FaceLocalId local_id) { _checkSameId(item,local_id); }
  void checkSameId(Cell item,CellLocalId local_id) { _checkSameId(item,local_id); }
  void checkSameId(Node item,NodeLocalId local_id) { _checkSameId(item,local_id); }

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  IMesh* m_mesh = nullptr;
  VariableCellReal3 m_cell_center;
  VariableFaceReal3 m_face_center;
  VariableNodeReal m_node_density; 
  Integer m_nb_print = 100;

 private:

  void _testDirCell();
  void _testDirFace();
  void _testDirNode();
  void _testDirCellNode();
  void _testDirCellFace();
  void _testDirFace(int idir);
  void _testNodeToCellConnectivity2D();
  void _testCellToNodeConnectivity2D();
  void _testNodeToCellConnectivity3D();
  void _testCellToNodeConnectivity3D();
  void _computeCenters();
  void _checkItemGroupIsSorted(const ItemGroup& group);

 private:

  void _sample(ICartesianMesh* cartesian_mesh);
  void _checkSameId(Face item,FaceLocalId local_id);
  void _checkSameId(Cell item,CellLocalId local_id);
  void _checkSameId(Node item,NodeLocalId local_id);
  void _saveSVG();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

