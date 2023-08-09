// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentModifierWorkInfo.h                                 (C) 2000-2023 */
/*                                                                           */
/* Structure de travail utilisée lors de la modification des constituants.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_COMPONENTMODIFIERWORKINFO_H
#define ARCANE_MATERIALS_INTERNAL_COMPONENTMODIFIERWORKINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure de travail utilisée lors de la modification
 * des constituants (via MeshMaterialModifier).
 *
 * Les instances de cette classe sont conservées durant toute une phase de
 * modification comprenant plusieurs operations de modification des matériaux
 * (MaterialModifierOperation).
 */
struct ARCANE_MATERIALS_EXPORT ComponentModifierWorkInfo
{
  using TransformCellsArgs = MeshMaterialVariableIndexer::TransformCellsArgs;

  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;
  bool is_verbose = false;
  bool is_add = false;

 public:

  void initialize(Int32 max_local_id);

 public:

  //! Indique si la maille \a local_id est transformée lors de l'opération courante.
  bool isTransformedCell(CellLocalId local_id) const
  {
    return m_cells_to_transform[local_id.localId()];
  }

  //! Positionne l'état de transformation de la maille \a local_id pour l'opération courante
  void setTransformedCell(CellLocalId local_id, bool v)
  {
    m_cells_to_transform[local_id.localId()] = v;
  }

  //! Positionne à \a value l'état 'Removed' des mailles de \a local_ids
  void setRemovedCells(ConstArrayView<Int32> local_ids, bool value);

 public:

  TransformCellsArgs toTransformCellsArgs()
  {
    return TransformCellsArgs(m_cells_to_transform, pure_local_ids,
                              partial_indexes, is_add, is_verbose);
  }

 public:

  // Filtre indiquant les mailles qui sont supprimées du constituant
  // Ce tableau est dimensionné au nombre de mailles.
  UniqueArray<bool> m_removed_local_ids_filter;

 private:

  // Filtre indiquant les mailles qui doivent changer de status (Pure<->Partial)
  // Ce tableau est dimensionné au nombre de mailles.
  UniqueArray<bool> m_cells_to_transform;
};

using IncrementalWorkInfo = ComponentModifierWorkInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
