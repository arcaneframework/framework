// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUniqueIdRenumbering.h                          (C) 2000-2021 */
/*                                                                           */
/* Renumérotation des uniqueId() pour les maillages cartésiens.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CEA_CARTESIANMESHUNIQUEIDRENUMBERING_H
#define ARCANE_CEA_CARTESIANMESHUNIQUEIDRENUMBERING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ItemTypes.h"
#include "arcane/VariableTypedef.h"

#include "arcane/cea/CeaGlobal.h"

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
  friend CartesianMesh;
 public:
  CartesianMeshUniqueIdRenumbering(ICartesianMesh* cmesh,ICartesianMeshGenerationInfo* gen_info);
  ~CartesianMeshUniqueIdRenumbering() = default;
 public:
  void renumber();
 private:
  ICartesianMesh* m_cartesian_mesh = nullptr;
  ICartesianMeshGenerationInfo* m_generation_info = nullptr;
  bool m_is_verbose = false;
 private:
  void _applyChildrenCell(Cell cell,VariableNodeInt64& nodes_new_uid,VariableFaceInt64& faces_new_uid,
                          VariableCellInt64& cells_new_uid,
                          Int64 coord_i,Int64 coord_j,
                          Int64 nb_cell_x,Int64 nb_cell_y,Int32 level);
  void _applyFamilyRenumbering(IItemFamily* family,VariableItemInt64& items_new_uid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
