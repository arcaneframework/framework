// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllEnvData.h                                                (C) 2000-2023 */
/*                                                                           */
/* Informations sur les valeurs des milieux.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_ALLENVDATA_H
#define ARCANE_MATERIALS_INTERNAL_ALLENVDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MatItemEnumerator.h"

#include "arcane/materials/internal/MeshMaterial.h"
#include "arcane/materials/internal/MeshEnvironment.h"
#include "arcane/materials/internal/ComponentItemInternalData.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialMng;
class MaterialModifierOperation;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations sur les valeurs des milieux.
 */
class AllEnvData
: public TraceAccessor
{
 public:

  explicit AllEnvData(MeshMaterialMng* mmg);

 public:

  void forceRecompute(bool compute_all);

 public:

  ArrayView<ComponentItemInternal> allEnvItemsInternal()
  {
    return m_item_internal_data.allEnvItemsInternal();
  }

  const VariableCellInt32& nbEnvPerCell() const
  {
    return m_nb_env_per_cell;
  }

  void updateMaterialDirect(MaterialModifierOperation* operation);

  //! Notification de la fin de création des milieux/matériaux
  void endCreate();

 private:

  MeshMaterialMng* m_material_mng = nullptr;

  //! Nombre de milieux par mailles
  VariableCellInt32 m_nb_env_per_cell;

  //! Niveau de verbosité
  Int32 m_verbose_debug_level = 0;

  ComponentItemInternalData m_item_internal_data;

 private:

  void _computeNbEnvAndNbMatPerCell();
  void _switchComponentItemsForEnvironments(const IMeshEnvironment* modified_env, bool is_add_operation);
  void _switchComponentItemsForMaterials(const MeshMaterial* modified_mat, bool is_add);
  void _copyBetweenPartialsAndGlobals(Int32ConstArrayView pure_local_ids,
                                      Int32ConstArrayView partial_indexes,
                                      Int32 indexer_index, bool is_add_operation);

  void _updateMaterialDirect(MaterialModifierOperation* operation);
  void _computeAndResizeEnvItemsInternal();
  bool _isFullVerbose() const;
  void _rebuildMaterialsAndEnvironmentsFromGroups();
  void _computeInfosForEnvCells();
  void _checkLocalIdsCoherency() const;
  void _printAllEnvCells(CellVectorView ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
