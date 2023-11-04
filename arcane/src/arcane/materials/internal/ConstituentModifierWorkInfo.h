// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentModifierWorkInfo.h                               (C) 2000-2023 */
/*                                                                           */
/* Structure de travail utilisée lors de la modification des constituants.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_CONSTITUENTMODIFIERWORKINFO_H
#define ARCANE_MATERIALS_INTERNAL_CONSTITUENTMODIFIERWORKINFO_H
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
 *
 * Il faut appeler initialize() avant d'utiliser l'instance.
 */
class ARCANE_MATERIALS_EXPORT ConstituentModifierWorkInfo
{
 public:

  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;
  bool is_verbose = false;

 public:

  //! Initialise l'instance.
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

  //! Indique si la maille \a local_id est supprimée du matériaux pour l'opération courante.
  bool isRemovedCell(Int32 local_id) const { return m_removed_local_ids_filter[local_id]; }

  //! Positionne à \a value l'état 'Removed' des mailles de \a local_ids
  void setRemovedCells(ConstArrayView<Int32> local_ids, bool value);

  //! Positionne l'opération courante
  void setCurrentOperation(MaterialModifierOperation* operation);

  //! Indique si l'opération courante est un ajout (true) ou une suppression (false) de mailles
  bool isAdd() const { return m_is_add; }

 private:

  // Filtre indiquant les mailles qui sont supprimées du constituant
  // Ce tableau est dimensionné au nombre de mailles.
  UniqueArray<bool> m_removed_local_ids_filter;

  // Filtre indiquant les mailles qui doivent changer de status (Pure<->Partial)
  // Ce tableau est dimensionné au nombre de mailles.
  UniqueArray<bool> m_cells_to_transform;

  bool m_is_add = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
