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
 * des constituants (via MeshMaterialModifier)
 */
struct ARCANE_MATERIALS_EXPORT ComponentModifierWorkInfo
{
  using TransformCellsArgs = MeshMaterialVariableIndexer::TransformCellsArgs;

  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;
  // Filtre indiquant les mailles qui doivent changer de status (Pure<->Partial)
  // Ce tableau est dimensionné au nombre de mailles.
  UniqueArray<bool> cells_to_transform;
  // Filtre indiquant les mailles qui sont supprimées du constituant
  // Ce tableau est dimensionné au nombre de mailles.
  UniqueArray<bool> removed_local_ids_filter;
  bool is_verbose = false;
  bool is_add = false;

 public:

  TransformCellsArgs toTransformCellsArgs()
  {
    return TransformCellsArgs(cells_to_transform, pure_local_ids,
                              partial_indexes, is_add, is_verbose);
  }
};

using IncrementalWorkInfo = ComponentModifierWorkInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
