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

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneL
{
class IVariableMng;
class Properties;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialMng;

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

  AllEnvData(MeshMaterialMng* mmg);
  virtual ~AllEnvData();

 public:

  void forceRecompute(bool compute_all);

 public:

  ConstArrayView<ComponentItemInternal> allEnvItemsInternal() const
  {
    return m_all_env_items_internal;
  }

  ArrayView<ComponentItemInternal> allEnvItemsInternal()
  {
    return m_all_env_items_internal;
  }

  const VariableCellInt32& nbEnvPerCell() const
  {
    return m_nb_env_per_cell;
  }

  void updateMaterialDirect(IMeshMaterial* mat,Int32ConstArrayView ids,eOperation operation);

  void printAllEnvCells(Int32ConstArrayView ids);

 private:

  MeshMaterialMng* m_material_mng = nullptr;

  /*!
   * \brief Infos sur les matériaux et les milieux.
   * Ce tableau est modifié chaque fois que les mailles des matériaux et
   * des milieux change.
   * Les premiers éléments de ce tableau contiennent
   * les infos pour les mailles de type AllEnvCell et peuvent
   * être indexés directement avec le localId() de ces mailles.
   */
  //@{
  VariableCellInt32 m_nb_env_per_cell;
  UniqueArray<ComponentItemInternal> m_all_env_items_internal;
  UniqueArray<ComponentItemInternal> m_env_items_internal;
  //@}

 private:

  void _computeNbEnvAndNbMatPerCell();
  void _switchComponentItemsForEnvironments(IMeshEnvironment* modified_env,eOperation add_or_remove);
  void _switchComponentItemsForMaterials(MeshMaterial* modified_mat,eOperation add_or_remove);
  Integer _checkMaterialPrescence(IMeshMaterial* mat,Int32ConstArrayView ids,
                                  eOperation operation);
  void _filterValidIds(IMeshMaterial* mat,Int32ConstArrayView ids,
                       bool do_add,Int32Array& valid_ids);
  void _copyBetweenPartialsAndGlobals(Int32ConstArrayView pure_local_ids,
                                      Int32ConstArrayView partial_indexes,
                                      Int32 indexer_index,eOperation operation);

  void _updateMaterialDirect(IMeshMaterial* mat,Int32ConstArrayView ids,eOperation add_or_remove);
  void _throwBadOperation(eOperation operation);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
