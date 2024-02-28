﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier.h                              (C) 2000-2024 */
/*                                                                           */
/* Modification incrémentale des constituants.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_INCREMENTALCOMPONENTMODIFIER_H
#define ARCANE_MATERIALS_INTERNAL_INCREMENTALCOMPONENTMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"
#include "arcane/materials/internal/ConstituentModifierWorkInfo.h"

#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Modification incrémentale des matériaux.
 *
 * Il faut appeler initialize() pour initialiser l'instance puis appeler
 * apply() pour chaque opération.
 */
class ARCANE_MATERIALS_EXPORT IncrementalComponentModifier
: public TraceAccessor
{
  friend class MeshMaterialModifierImpl;

 public:

  explicit IncrementalComponentModifier(AllEnvData* all_env_data);

 public:

  void initialize();
  void apply(MaterialModifierOperation* operation);
  void finalize();

 private:

  AllEnvData* m_all_env_data = nullptr;
  MeshMaterialMng* m_material_mng = nullptr;
  ConstituentModifierWorkInfo m_work_info;
  RunQueue m_copy_queue;

 private:

  void _switchCellsForEnvironments(const IMeshEnvironment* modified_env,
                                   ConstArrayView<Int32> ids);
  void _switchCellsForMaterials(const MeshMaterial* modified_mat,
                                ConstArrayView<Int32> ids);
  void _computeCellsToTransformForMaterial(const MeshMaterial* mat, ConstArrayView<Int32> ids);
  void _computeCellsToTransformForEnvironments(ConstArrayView<Int32> ids);
  void _removeItemsFromEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                                   Int32ConstArrayView local_ids, bool update_env_indexer);
  void _addItemsToEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                              Int32ConstArrayView local_ids, bool update_env_indexer);
  void _addItemsToIndexer(MeshEnvironment* env, MeshMaterialVariableIndexer* var_indexer,
                          Int32ConstArrayView local_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
