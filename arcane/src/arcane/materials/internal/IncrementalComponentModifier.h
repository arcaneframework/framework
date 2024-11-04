// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
class CopyBetweenPartialAndGlobalArgs;

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

  IncrementalComponentModifier(AllEnvData* all_env_data,const RunQueue& queue);

 public:

  void initialize(bool is_debug);
  void apply(MaterialModifierOperation* operation);
  void finalize();
  void setDoCopyBetweenPartialAndPure(bool v) { m_do_copy_between_partial_and_pure = v; }
  void setDoInitNewItems(bool v) { m_do_init_new_items = v; }

 private:

  AllEnvData* m_all_env_data = nullptr;
  MeshMaterialMng* m_material_mng = nullptr;
  ConstituentModifierWorkInfo m_work_info;
  RunQueue m_queue;
  bool m_do_copy_between_partial_and_pure = true;
  bool m_do_init_new_items = true;
  bool m_is_debug = false;
  //! 1 ou 2 si on utilise une version générique pour les copies entre pure et partiel
  Int32 m_use_generic_copy_between_pure_and_partial = 0;

 public:

  void flagRemovedCells(SmallSpan<const Int32> local_ids, bool value_to_set);

 public:

  Int32 _computeCellsToTransformForEnvironments(SmallSpan<const Int32> ids);
  void _resetTransformedCells(SmallSpan<const Int32> ids);
  void _addItemsToIndexer(MeshMaterialVariableIndexer* var_indexer,
                          SmallSpan<const Int32> local_ids);
  void _removeItemsInGroup(ItemGroup cells,SmallSpan<const Int32> removed_ids);
  void _applyCopyBetweenPartialsAndGlobals(const CopyBetweenPartialAndGlobalArgs& args, RunQueue& queue);

 private:

  void _switchCellsForEnvironments(const IMeshEnvironment* modified_env,
                                   SmallSpan<const Int32> ids);
  void _switchCellsForMaterials(const MeshMaterial* modified_mat,
                                SmallSpan<const Int32> ids);
  Int32 _computeCellsToTransformForMaterial(const MeshMaterial* mat, SmallSpan<const Int32> ids);
  void _removeItemsFromEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                                   SmallSpan<const Int32> local_ids, bool update_env_indexer);
  void _addItemsToEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                              SmallSpan<const Int32> local_ids, bool update_env_indexer);
  void _copyBetweenPartialsAndGlobals(const CopyBetweenPartialAndGlobalArgs& args);
  void _resizeVariablesIndexer(Int32 var_index);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
