// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifierImpl.h                                  (C) 2000-2024 */
/*                                                                           */
/* Implémentation de la modification des matériaux et milieux.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALMODIFIERIMPL_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALMODIFIERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/internal/IncrementalComponentModifier.h"

#include "arcane/accelerator/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialMng;
class MaterialModifierOperation;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialModifierImpl
: public TraceAccessor
{
 private:
  
  using Operation = MaterialModifierOperation;

  class OperationList
  {
   public:
    ~OperationList();
   public:
    void add(Operation* o);
    void clear();
    ConstArrayView<Operation*> values() const { return m_operations.constView(); }
   public:
    UniqueArray<Operation*> m_operations;
  };

 public:

  explicit MeshMaterialModifierImpl(MeshMaterialMng* mm);

 public:

  void initOptimizationFlags();
  void setDoCopyBetweenPartialAndPure(bool v) { m_do_copy_between_partial_and_pure = v; }
  void setDoInitNewItems(bool v) { m_do_init_new_items = v; }
  void setPersistantWorkBuffer(bool v) { m_is_keep_work_buffer = v; }

 public:

  void reset();

  void addCells(IMeshMaterial* mat, SmallSpan<const Int32> ids);
  void removeCells(IMeshMaterial* mat, SmallSpan<const Int32> ids);

  void endUpdate();
  void beginUpdate();
  void dumpStats();

 private:

  void _addCellsToGroupDirect(IMeshMaterial* mat, SmallSpan<const Int32> ids);
  void _removeCellsToGroupDirect(IMeshMaterial* mat, SmallSpan<const Int32> ids);

  void _applyOperationsNoOptimize();
  void _updateEnvironmentsNoOptimize();
  bool _checkMayOptimize();

 private:

  MeshMaterialMng* m_material_mng = nullptr;
  OperationList m_operations;
  RunQueue m_queue;
  std::unique_ptr<IncrementalComponentModifier> m_incremental_modifier;

  Int32 nb_update = 0;
  Int32 nb_save_restore = 0;
  Int32 nb_optimize_add = 0;
  Int32 nb_optimize_remove = 0;
  Int32 m_modification_id = 0;

  bool m_allow_optimization = false;
  bool m_allow_optimize_multiple_operation = false;
  bool m_allow_optimize_multiple_material = false;
  bool m_use_incremental_recompute = false;
  bool m_print_component_list = false;

  bool m_do_copy_between_partial_and_pure = true;
  bool m_do_init_new_items = true;
  bool m_is_keep_work_buffer = true;

 private:

  void _endUpdate();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

