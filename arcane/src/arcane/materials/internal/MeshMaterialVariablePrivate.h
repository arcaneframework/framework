// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariablePrivate.h                               (C) 2000-2024 */
/*                                                                           */
/* Private section of a variable on a mesh material.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALVARIABLEPRIVATE_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALVARIABLEPRIVATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/VariableDependInfo.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"

#include "arcane/materials/MeshMaterialVariableDependInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IObserver;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Private section of a material variable.
 */
class MeshMaterialVariablePrivate
: public IMeshMaterialVariableInternal
{
 public:

  MeshMaterialVariablePrivate(const MaterialVariableBuildInfo& v, MatVarSpace mvs,
                              MeshMaterialVariable* variable);
  ~MeshMaterialVariablePrivate();

 public:

  MatVarSpace space() const { return m_var_space; }
  bool hasRecursiveDepend() const { return m_has_recursive_depend; }
  const String& name() const { return m_name; }
  IMeshMaterialMng* materialMng() const { return m_material_mng; }
  IMeshMaterialVariableInternal* _internalApi() { return this; }

 public:

  Int32 dataTypeSize() const override;
  void copyToBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
                    Span<std::byte> bytes, RunQueue* queue) const override;

  void copyFromBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
                      Span<const std::byte> bytes, RunQueue* queue) override;

  Ref<IData> internalCreateSaveDataRef(Integer nb_value) override;

  void saveData(IMeshComponent* component,IData* data) override;

  void restoreData(IMeshComponent* component, IData* data, Integer data_index,
                   Int32ConstArrayView ids, bool allow_null_id) override;

  void copyBetweenPartialAndGlobal(const CopyBetweenPartialAndGlobalArgs& args) override;

  void initializeNewItemsWithZero(InitializeWithZeroArgs& args) override;

  ConstArrayView<VariableRef*> variableReferenceList() const override
  {
    return m_refs.view();
  }
  void syncReferences(bool check_resize) override;
  void resizeForIndexer(ResizeVariableIndexerArgs& args) override;

 public:

  Int32 m_nb_reference = 0;
  MeshMaterialVariableRef* m_first_reference = nullptr; //! First reference on the variable

 private:

  String m_name;
  IMeshMaterialMng* m_material_mng = nullptr;

 public:

  /*!
   * \brief Stores references to array variables used to
   * store values per material.
   * A reference must be kept to prevent the variable from being destroyed
   * if it is no longer used elsewhere.
   */
  UniqueArray<VariableRef*> m_refs;

  bool m_keep_on_change = true;
  IObserver* m_global_variable_changed_observer = nullptr;

  //! List of dependencies for this variable
  UniqueArray<MeshMaterialVariableDependInfo> m_mat_depends;

  //! List of dependencies for this variable
  UniqueArray<VariableDependInfo> m_depends;

  //! Tag of the last modification per material
  UniqueArray<Int64> m_modified_times;

  //! Calculation function
  ScopedPtrT<IMeshMaterialVariableComputeFunction> m_compute_function;

 private:

  bool m_has_recursive_depend = true;
  MatVarSpace m_var_space = MatVarSpace::MaterialAndEnvironment;
  MeshMaterialVariable* m_variable = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
