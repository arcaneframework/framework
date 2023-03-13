// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshChecker.h                                        (C) 2000-2018 */
/*                                                                           */
/* Classe fournissant des méthodes de vérification sur le maillage.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DYNAMICMESHCHECKER_H
#define ARCANE_MESH_DYNAMICMESHCHECKER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/VariableTypedef.h"
#include "arcane/IMeshChecker.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

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

  DynamicMeshChecker(IMesh* mesh);

  ~DynamicMeshChecker();

 public:

  IMesh* mesh() override
  {
    return m_mesh;
  }

  void setCheckLevel(Integer level) override
  {
    m_check_level = level;
  }

  Integer checkLevel() const override
  {
    return m_check_level;
  }

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
  void updateAMRFaceOrientation(ArrayView<Int64> ghost_cell_to_refine) ;

 private:

  void _checkFacesOrientation();
  void _checkValidItemOwner(IItemFamily* family);
  void _checkReplicationFamily(IItemFamily* family);

 private:

  IMesh* m_mesh;
  Integer m_check_level;

  VariableCellArrayInt64* m_var_cells_faces;
  VariableCellArrayInt64* m_var_cells_nodes;

  bool m_compare_reference_file;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_DYNAMICMESHCHECKER_H */
