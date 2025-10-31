// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizer.h                          (C) 2000-2025 */
/*                                                                           */
/* Synchroniseur de variables matériaux.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALVARIABLESYNCHRONIZER_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

#include "arcane/materials/MatVarIndex.h"
#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllEnvCellVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Synchroniseur de variables matériaux.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableSynchronizer
: public TraceAccessor
, public IMeshMaterialVariableSynchronizer
{
 public:

  MeshMaterialVariableSynchronizer(IMeshMaterialMng* material_mng,
                                   IVariableSynchronizer* var_syncer,
                                   MatVarSpace mvs);

 public:

  IVariableSynchronizer* variableSynchronizer() override;
  ConstArrayView<MatVarIndex> sharedItems(Int32 index) override;
  ConstArrayView<MatVarIndex> ghostItems(Int32 index) override;
  void recompute() override;
  void checkRecompute() override;
  Ref<IMeshMaterialSynchronizeBuffer> commonBuffer() override { return m_common_buffer; }
  eMemoryRessource bufferMemoryRessource() const override { return m_buffer_memory_ressource; }

 private:

  IMeshMaterialMng* m_material_mng;
  IVariableSynchronizer* m_variable_synchronizer;
  UniqueArray<UniqueArray<MatVarIndex>> m_shared_items;
  UniqueArray<UniqueArray<MatVarIndex>> m_ghost_items;
  Int64 m_timestamp;
  MatVarSpace m_var_space;
  Ref<IMeshMaterialSynchronizeBuffer> m_common_buffer;
  eMemoryRessource m_buffer_memory_ressource = eMemoryRessource::UnifiedMemory;
  // Permet de forcer l'utilisation ou non l'implémentation accélérateur
  Int32 m_use_accelerator_mode = -1;

 public:

  // Doit être publique pour CUDA.
  void _fillCellsAccelerator(Array<MatVarIndex>& items, AllEnvCellVectorView view, RunQueue& queue);

 private:

  void _fillCells(Array<MatVarIndex>& items, AllEnvCellVectorView view, RunQueue& queue);
  void _fillCellsSequential(Array<MatVarIndex>& items, AllEnvCellVectorView view);
  void _initialize();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

