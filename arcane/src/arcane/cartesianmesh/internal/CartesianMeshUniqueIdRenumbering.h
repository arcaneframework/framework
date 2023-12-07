// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUniqueIdRenumbering.h                          (C) 2000-2023 */
/*                                                                           */
/* Renumérotation des uniqueId() pour les maillages cartésiens.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_CARTESIANMESHUNIQUEIDRENUMBERING_H
#define ARCANE_CARTESIANMESH_INTERNAL_CARTESIANMESHUNIQUEIDRENUMBERING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"

#include "arcane/cartesianmesh/CartesianPatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ICartesianMeshGenerationInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Renumérotation des uniqueId() pour les maillages cartésiens.
 *
 * Renumérote les uniqueId() des noeuds, faces et mailles pour avoir la même
 * numérotation en séquentiel et parallèle.
 */
class CartesianMeshUniqueIdRenumbering
: public TraceAccessor
{
  class NewUniqueIdList;

 public:

  CartesianMeshUniqueIdRenumbering(ICartesianMesh* cmesh, ICartesianMeshGenerationInfo* gen_info,
                                   CartesianPatch parent_patch);
  ~CartesianMeshUniqueIdRenumbering() = default;

 public:

  void renumber();

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  ICartesianMeshGenerationInfo* m_generation_info = nullptr;
  CartesianPatch m_parent_patch;
  bool m_is_verbose = false;

 private:

  void _applyChildrenCell2D(Cell cell, NewUniqueIdList& new_uids,
                            Int64 coord_i, Int64 coord_j,
                            Int64 nb_cell_x, Int64 nb_cell_y, Int32 level, Int64 base_adder);
  void _applyChildrenCell3D(Cell cell, NewUniqueIdList& new_uids,
                            Int64 coord_i, Int64 coord_j, Int64 coord_k,
                            Int64 current_level_nb_cell_x, Int64 current_level_nb_cell_y, Int64 current_level_nb_cell_z,
                            Int32 current_level, Int64 cell_adder, Int64 node_adder, Int64 face_adder);
  void _applyFamilyRenumbering(IItemFamily* family, VariableItemInt64& items_new_uid);
  void _markItemsToKeepCurrentNumbering(ICartesianMeshPatch* patch);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
