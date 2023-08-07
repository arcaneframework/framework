// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier.h                              (C) 2000-2023 */
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class AllEnvData;
class MaterialModifierOperation;
class MeshMaterial;
class MeshEnvironment;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure de travail utilisée lors de la modification
 * des constituants (via MeshMaterialModifier)
 */
struct ARCANE_MATERIALS_EXPORT IncrementalWorkInfo
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

 private:

  AllEnvData* m_all_env_data = nullptr;
  MeshMaterialMng* m_material_mng = nullptr;
  IncrementalWorkInfo m_work_info;

 private:

  void _switchComponentItemsForEnvironments(const IMeshEnvironment* modified_env);
  void _switchComponentItemsForMaterials(const MeshMaterial* modified_mat);
  void _computeCellsToTransform(const MeshMaterial* mat);
  void _computeCellsToTransform();
  void _removeItemsFromEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                                   Int32ConstArrayView local_ids, bool update_env_indexer);
  void _addItemsToEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                              Int32ConstArrayView local_ids, bool update_env_indexer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
