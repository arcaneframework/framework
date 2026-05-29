// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephTest.h                                                      (C) 2011 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_TEST_H
#define ALEPH_TEST_H

#include <set>

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/ITiedInterface.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/ItemArrayEnumerator.h"
#include "arcane/core/ItemPairEnumerator.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/BasicModule.h"
#include "arcane/core/SharedVariable.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/ItemRefinement.h"
#include "arcane/mesh/MeshRefinement.h"
#include "arcane/mesh/FaceReorienter.h"

#include "arcane/aleph/AlephTypesSolver.h"
#include "arcane/aleph/AlephArcane.h"

#include "arcane/aleph/IAleph.h"
#include "arcane/aleph/IAlephFactory.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#define ARCANE_ENABLE_AMR
#ifdef ARCANE_ENABLE_AMR

#define MESH_ALL_ACTIVE_CELLS(mesh) mesh->allActiveCells()
#define MESH_OWN_ACTIVE_CELLS(mesh) mesh->ownActiveCells()
#define CELL_NB_H_CHILDREN(iCell) iCell.nbHChildren()
#define CELL_HAS_H_CHILDREN(iCell) iCell.hasHChildren()
#define CELL_H_CHILD(iCell, j) iCell.hChild(j).toCell()
#define CELL_H_PARENT(iCell) iCell.hParent()
#define INNER_ACTIVE_FACE_GROUP(cells) cells.innerActiveFaceGroup()
#define OUTER_ACTIVE_FACE_GROUP(cells) cells.outerActiveFaceGroup()
#define MESH_MODIFIER_REFINE_ITEMS(mesh) mesh->modifier()->refineItems()
#define MESH_MODIFIER_CORSEN_ITEMS(mesh) mesh->modifier()->coarsenItems()

#else /* ARCANE_ENABLE_AMR */

#define MESH_ALL_ACTIVE_CELLS(mesh) mesh->allCells()
#define MESH_OWN_ACTIVE_CELLS(mesh) mesh->ownCells()
#define CELL_NB_H_CHILDREN(iCell) 0
#define CELL_HAS_H_CHILDREN(iCell) false
#define CELL_H_CHILD(iCell, j) (*iCell)
#define CELL_H_PARENT(iCell) (*iCell)
#define INNER_ACTIVE_FACE_GROUP(cells) cells.innerFaceGroup()
#define OUTER_ACTIVE_FACE_GROUP(cells) cells.outerFaceGroup()
#define MESH_MODIFIER_REFINE_ITEMS(mesh)
#define MESH_MODIFIER_COARSEN_ITEMS(mesh)

#endif /* ARCANE_ENABLE_AMR */

#define ECART_RELATIF(T0, T1) (math::abs((T0 - T1) / (T0 + T1)))

#include "arcane/aleph/tests/AlephTestScheme.h"
//#include "arcane/aleph/tests/AlephTestSchemeFaces.h"
#include "arcane/aleph/tests/AlephTestModule.h"

#endif // ALEPH_TEST_H
