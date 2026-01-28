// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshTestUtils.cc                                   (C) 2000-2026 */
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

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

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

  explicit CartesianMeshTestUtils(ICartesianMesh* cm, Accelerator::IAcceleratorMng* am);
  ~CartesianMeshTestUtils();

 public:

  void testAll(bool is_amr);

 public:

  void checkSameId(Face item, FaceLocalId local_id) { _checkSameId(item, local_id); }
  void checkSameId(Cell item, CellLocalId local_id) { _checkSameId(item, local_id); }
  void checkSameId(Node item,NodeLocalId local_id) { _checkSameId(item,local_id); }
  void setNbPrint(Int32 v) { m_nb_print = v; }

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  IMesh* m_mesh = nullptr;
  Accelerator::IAcceleratorMng* m_accelerator_mng = nullptr;
  VariableCellReal3 m_cell_center;
  VariableFaceReal3 m_face_center;
  VariableNodeReal m_node_density; 
  Int32 m_nb_print = 100;
  bool m_is_amr = false;

 private:

  void _testDirCell();
  void _testDirFace();
  void _testDirFaceAccelerator();
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
  void _checkSameId(FaceLocalId item, FaceLocalId local_id);
  void _checkSameId(CellLocalId item, CellLocalId local_id);
  void _checkSameId(NodeLocalId item, NodeLocalId local_id);
  void _saveSVG();

 public:

  //! Méthodes publiques car accessibles sur accélérateur
  void _testDirCellAccelerator();
  void _testDirFaceAccelerator(int idir);
  void _testDirCellNodeAccelerator();
  void _testDirNodeAccelerator();
  void _testDirCellFaceAccelerator();
  void _testNodeToCellConnectivity3DAccelerator();
  void _testCellToNodeConnectivity3DAccelerator();
  void _testConnectivityByDirection();
  template<typename ItemType> void
  _testConnectivityByDirectionHelper(const ItemGroup& group);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

