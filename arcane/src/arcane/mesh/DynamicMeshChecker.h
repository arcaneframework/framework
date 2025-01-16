// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshChecker.h                                        (C) 2000-2025 */
/*                                                                           */
/* Classe fournissant des méthodes de vérification sur le maillage.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DYNAMICMESHCHECKER_H
#define ARCANE_MESH_DYNAMICMESHCHECKER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/VariableTypedef.h"
#include "arcane/core/IMeshChecker.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;
class ItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMeshChecker
: public TraceAccessor
, public IMeshChecker
{
 public:

  explicit DynamicMeshChecker(IMesh* mesh);
  ~DynamicMeshChecker();

 public:

  IMesh* mesh() override { return m_mesh; }
  void setCheckLevel(Integer level) override { m_check_level = level; }
  Integer checkLevel() const override { return m_check_level; }

  void checkValidMesh() override;
  void checkValidMeshFull() override;
  void checkValidReplication() override;
  void checkVariablesSynchronization() override;
  void checkItemGroupsSynchronization() override;

 public:

  void checkValidConnectivity();
  void checkGhostCells();
  void checkMeshFromReferenceFile();
  void updateAMRFaceOrientation();
  void updateAMRFaceOrientation(ArrayView<Int64> ghost_cell_to_refine);

 private:

  void _checkFacesOrientation();
  void _checkValidItemOwner(IItemFamily* family);
  void _checkReplicationFamily(IItemFamily* family);

 private:

  IMesh* m_mesh = nullptr;
  Integer m_check_level = 0;

  VariableCellArrayInt64* m_var_cells_faces = nullptr;
  VariableCellArrayInt64* m_var_cells_nodes = nullptr;

  bool m_compare_reference_file = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_DYNAMICMESHCHECKER_H */
