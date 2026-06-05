// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemMaterialVariableBaseT.h                                 (C) 2000-2024 */
/*                                                                           */
/* Implementation of the base class for material variables.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Mutex.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/Observer.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshMaterialVariableFactoryMng.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"
#include "arcane/materials/internal/MeshMaterialVariablePrivate.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

namespace
{

  // Allocator used for views
  // We use accelerator memory if the execution is done on an accelerator
  MemoryAllocationOptions
  _getViewAllocator(IMeshMaterialMng* mm)
  {
    eMemoryRessource r = eMemoryRessource::Host;
    if (mm->_internalApi()->runQueue().isAcceleratorPolicy())
      r = eMemoryRessource::Device;
    return MemoryUtils::getAllocationOptions(r);
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits>
ItemMaterialVariableBase<Traits>::
ItemMaterialVariableBase(const MaterialVariableBuildInfo& v,
                         PrivatePartType* global_var,
                         VariableRef* global_var_ref, MatVarSpace mvs)
: MeshMaterialVariable(v, mvs)
, m_global_variable(global_var)
, m_global_variable_ref(global_var_ref)
, m_device_views(_getViewAllocator(m_p->materialMng()))
, m_host_views(MemoryUtils::getAllocationOptions(eMemoryRessource::HostPinned))
{
  m_device_views.setDebugName(String("ItemMaterialVariableViews") + v.name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits>
ItemMaterialVariableBase<Traits>::
~ItemMaterialVariableBase()
{
  Integer nb_var = m_p->m_refs.size();
  delete m_global_variable_ref;
  for (Integer i = 1; i < nb_var; ++i) {
    delete m_p->m_refs[i];
  }
  m_p->m_refs.clear();
  m_vars.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> IVariable*
ItemMaterialVariableBase<Traits>::
globalVariable() const
{
  return m_global_variable;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_init(ArrayView<PrivatePartType*> vars)
{
  if (!m_vars.empty())
    ARCANE_FATAL("Internal error: _init() already called");
  m_vars.addRange(vars);
  Integer nb_var = m_vars.size();
  m_device_views.resizeNoInit(nb_var);
  m_host_views.resize(nb_var);
  m_views_as_bytes.resize(nb_var);
  // We must maintain a reference to the variables we possess to
  // ensure they are properly initialized and will not be destroyed
  // as long as this instance exists.
  m_p->m_refs.resize(nb_var);
  for (Integer i = 0; i < nb_var; ++i) {
    // m_vars[i] can be null if our variable is a medium-only variable.
    m_p->m_refs[i] = nullptr;
    if (i == 0) {
      // For the global variable, it is a mesh variable.
      m_p->m_refs[i] = m_global_variable_ref;
    }
    else {
      // For the others, it is an array variable.
      if (m_vars[i])
        m_p->m_refs[i] = new VariableRefType(m_vars[i]);
    }
    // TODO: we must be able to be notified of this variable's change.
    // to then update the views (maybe also create a MyVariableRef).
    _setView(i);
    //std::cout << "INIT_VIEW: " << i << " v=" << m_views[i].size() << '\n';
  }
  _copyHostViewsToViews(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
buildFromManager(bool is_continue)
{
  IMeshMaterialMng* mat_mng = m_p->materialMng();

  // We must retrieve (or create) the other variables.
  //ConstArrayView<IMeshMaterial*> materials = mat_mng->materials();
  IMeshMaterialMngInternal* mm_api = mat_mng->_internalApi();
  const Real reserve_ratio = mm_api->additionalCapacityRatio();
  ConstArrayView<MeshMaterialVariableIndexer*> indexers = mm_api->variablesIndexer();
  Integer nb_indexer = indexers.size();
  UniqueArray<PrivatePartType*> all_vars(nb_indexer + 1);
  all_vars[0] = m_global_variable;
  IMesh* mesh = m_global_variable->mesh();

  if (!m_global_variable->isUsed())
    m_global_variable->setUsed(true);

  // We must attach an observer on the global variable to know when it changes.
  // For partial variables, this is done automatically when the associated material changes.
  m_p->m_global_variable_changed_observer = new ObserverT<ThatClass>(this, &ThatClass::_syncFromGlobalVariable);
  m_global_variable->onSizeChangedObservable()->attachObserver(m_p->m_global_variable_changed_observer);

  m_p->m_modified_times.resize(nb_indexer);
  m_p->m_modified_times.fill(0);

  bool is_env_only = m_p->space() == MatVarSpace::Environment;
  //TODO: check if we need to store this variable's properties
  // before its creation
  int property = m_global_variable->property();
  for (Integer i = 0; i < nb_indexer; ++i) {
    MeshMaterialVariableIndexer* indexer = indexers[i];
    // Does nothing if we are only a medium variable and the indexer
    // is not associated with a medium.
    if (is_env_only && !indexer->isEnvironment()) {
      all_vars[i + 1] = nullptr;
      continue;
    }

    // If a name is specified, it is the compatibility mode
    // with Troll and in this case the variable is not associated
    // with the mesh.
    String var_name = String("Mat") + indexer->name() + "_" + this->name();

    // Note: These variables are not associated with the mesh.
    // They are just array variables.
    IVariable* var = nullptr;
    {
      VariableBuildInfo vbi2(mesh, var_name, property);
      VariableInfo vi = VariableRefType::_internalVariableInfo(vbi2);
      var = PrivatePartType::getReference(vbi2, vi);
    }
    // Tag the variable for codes that would want to know that it
    // is a material variable.
    var->addTag("Material", "1");
    PrivatePartType* true_ptr = dynamic_cast<PrivatePartType*>(var);
    ARCANE_CHECK_POINTER(true_ptr);
    all_vars[i + 1] = true_ptr;
    if (!is_continue) {
      // TODO: check if setUsed() is necessary.
      true_ptr->setUsed(true);
      // Resize the variable to the size of the mixed values array.
      Traits::resizeWithReserve(true_ptr, indexer->maxIndexInMultipleArray(), reserve_ratio);
    }
  }
  this->_init(all_vars);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_syncFromGlobalVariable()
{
  // This method is called when the global variable changes its size.
  // In this case, only the global variable's view must be modified.
  // But this method can also be called when the global variable's setUsed()
  // method is called and in this case all views must be updated. Since currently
  // we cannot distinguish the two
  // calls, we perform a complete update in all cases.
  const bool do_complete_sync = true;
  if (do_complete_sync) {
    syncReferences();
    return;
  }

  // This method is called when the global variable changes its size.
  // This variable is the first variable in the list.
  // We must then notify all references of this change.
  _setView(0);
  _copyHostViewsToViews(nullptr);

  for (MeshMaterialVariableRef::Enumerator i(this); i.hasNext(); ++i) {
    MeshMaterialVariableRef* ref = *i;
    ref->updateFromInternal();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
syncReferences()
{
  _syncReferences(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_syncReferences(bool check_resize)
{
  if (check_resize) {
    IMeshMaterialMngInternal* api = m_p->materialMng()->_internalApi();
    ConstArrayView<MeshMaterialVariableIndexer*> indexers = api->variablesIndexer();
    Real ratio = api->additionalCapacityRatio();
    Integer nb_indexer = indexers.size();
    for (Integer i = 0; i < nb_indexer; ++i) {
      MeshMaterialVariableIndexer* indexer = indexers[i];
      PrivatePartType* true_ptr = m_vars[i + 1];
      if (true_ptr) {
        Traits::resizeWithReserve(true_ptr, indexer->maxIndexInMultipleArray(), ratio);
        _setView(i + 1);
      }
    }
    _setView(0);
    _copyHostViewsToViews(nullptr);
  }

  for (MeshMaterialVariableRef::Enumerator i(this); i.hasNext(); ++i) {
    MeshMaterialVariableRef* ref = *i;
    ref->updateFromInternal();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> Ref<IData>
ItemMaterialVariableBase<Traits>::
_internalCreateSaveDataRef(Integer nb_value)
{
  Ref<IData> data = m_global_variable->data()->cloneEmptyRef();
  auto* true_data = dynamic_cast<ValueDataType*>(data.get());
  ARCANE_CHECK_POINTER(true_data);
  true_data->_internal()->reserve(nb_value);
  return data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_saveData(IMeshComponent* component, IData* data)
{
  if (m_p->space() == MatVarSpace::Environment && !component->isEnvironment())
    ARCANE_FATAL("Can not save data on material for environment only variable");
  Traits::saveData(component, data, m_host_views);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_restoreData(IMeshComponent* component, IData* data, Integer data_index,
             Int32ConstArrayView ids, bool allow_null_id)
{
  if (m_p->space() == MatVarSpace::Environment && !component->isEnvironment())
    ARCANE_FATAL("Can not restore data on material for environment only variable");

  auto* true_data = dynamic_cast<ValueDataType*>(data);
  ARCANE_CHECK_POINTER(true_data);
  auto values = true_data->view();

  PrivatePartType* global_var = this->_trueGlobalVariable();

  // Array to initialize values either to zero or with
  // the global variable's value.
  UniqueContainerType tmp_values;
  if (component->materialMng()->isDataInitialisationWithZero()) {
    Integer max_id = m_global_variable->itemFamily()->maxLocalId();
    Traits::resizeAndFillWithDefault(true_data, tmp_values, max_id);
  }
  else
    tmp_values = global_var->constValueView();

  if (allow_null_id) {
    for (Integer i = 0, n = ids.size(); i < n; ++i) {
      if (ids[i] != NULL_ITEM_ID) {
        Traits::setValue(tmp_values[ids[i]], values[data_index + i]);
      }
    }
  }
  else {
    for (Integer i = 0, n = ids.size(); i < n; ++i) {
      Traits::setValue(tmp_values[ids[i]], values[data_index + i]);
    }
  }

  ENUMERATE_COMPONENTCELL (icell, component) {
    ComponentCell ec = *icell;
    setValue(ec._varIndex(), tmp_values[ec.globalCell().localId()]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> bool
ItemMaterialVariableBase<Traits>::
_isValidAndUsedAndGlobalUsed(PrivatePartType* partial_var)
{
  if (!partial_var)
    return false;
  if (!m_global_variable->isUsed() || !partial_var->isUsed())
    return false;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_resizeForIndexer(ResizeVariableIndexerArgs& args)
{
  Int32 index = args.m_var_index;
  RunQueue& queue = args.m_queue;
  _setView(0);
  PrivatePartType* partial_var = m_vars[index + 1];
  if (_isValidAndUsedAndGlobalUsed(partial_var)) {
    // Resizes the multiple values array if necessary.
    // TODO: Perform on the device for view update
    // TODO: Use asynchronous allocation if possible (if no initialization)
    // TODO: Be able to indicate that we do not want to initialize the variable
    //       regardless of the value of eDataInitialisationPolicy
    IMeshMaterialMngInternal* api = m_p->materialMng()->_internalApi();
    ConstArrayView<MeshMaterialVariableIndexer*> indexers = api->variablesIndexer();
    Real ratio = api->additionalCapacityRatio();
    Traits::resizeWithReserve(partial_var, indexers[index]->maxIndexInMultipleArray(), ratio);
    this->_setView(index + 1);
  }
  if (args.isUseOneCommand()) {
    auto dest = asWritableBytes(m_device_views.view());
    auto source = asBytes(m_host_views.view());
    args.addOneCopyData(source, dest, 1);
  }
  else
    _copyHostViewsToViews(&queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_copyHostViewsToViews(RunQueue* queue)
{
  auto dest = asWritableBytes(m_device_views);
  auto source = asBytes(m_host_views);
  if (queue) {
    Accelerator::MemoryCopyArgs copy_args(dest, source);
    copy_args.addAsync(true);
    queue->copyMemory(copy_args);
  }
  else
    MemoryUtils::copy(dest, source);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_copyBetweenPartialAndGlobal(const CopyBetweenPartialAndGlobalArgs& args)
{
  Int32 var_index = args.m_var_index + 1;
  PrivatePartType* partial_var = m_vars[var_index];
  if (!_isValidAndUsedAndGlobalUsed(partial_var))
    return;
  const bool is_global_to_partial = args.m_is_global_to_partial;
  const bool use_generic = args.m_use_generic_copy;
  const RunQueue& queue = args.m_queue;

  auto input = m_host_views[var_index];
  auto output = m_host_views[0];
  auto output_indexes = args.m_local_ids;
  auto input_indexes = args.m_indexes_in_multiple;

  if (is_global_to_partial) {
    std::swap(input, output);
    std::swap(output_indexes, input_indexes);
  }
  if (use_generic) {
    Int32 data_type_size = dataTypeSize();
    SmallSpan<const std::byte> input_bytes(Traits::toBytes(input));
    SmallSpan<std::byte> output_bytes(Traits::toBytes(output));
    if (args.isUseOneCommand())
      args.addOneCopyData(input_bytes, output_bytes, data_type_size);
    else
      _genericCopyTo(input_bytes, input_indexes,
                     output_bytes, output_indexes, queue, data_type_size);
  }
  else {
    Traits::copyTo(input, input_indexes, output, output_indexes, queue);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_initializeNewItemsWithZero(InitializeWithZeroArgs& args)
{
  const Int32 var_index = args.m_var_index + 1;
  RunQueue& queue = args.m_queue;
  PrivatePartType* partial_var = m_vars[var_index];
  if (!_isValidAndUsedAndGlobalUsed(partial_var))
    return;

  SmallSpan<const Int32> partial_indexes = args.m_indexes_in_multiple;

  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, partial_indexes.data());

  if (args.isUseOneCommand()) {
    Int32 data_type_size = dataTypeSize();
    auto output = m_host_views[var_index];
    SmallSpan<std::byte> output_bytes(Traits::toBytes(output));
    args.addOneCopyData({}, output_bytes, data_type_size);
  }
  else {
    ContainerSpanType partial_view = m_host_views[var_index];
    Int32 nb_partial = partial_indexes.size();
    DataType zero = DataType();
    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, nb_partial)
    {
      auto [i] = iter();
      Int32 index = partial_indexes[i];
      Traits::setValue(partial_view[index], zero);
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
fillPartialValuesWithGlobalValues()
{
  fillPartialValuesWithSuperValues(LEVEL_ALLENVIRONMENT);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
fillPartialValuesWithSuperValues(Int32 level)
{
  IMeshMaterialMng* mm = m_p->materialMng();
  if (level == LEVEL_MATERIAL)
    _fillPartialValuesWithSuperValues(mm->materialsAsComponents());
  else if (level == LEVEL_ENVIRONMENT) {
    _fillPartialValuesWithSuperValues(mm->environmentsAsComponents());
  }
  else if (level == LEVEL_ALLENVIRONMENT) {
    _fillPartialValuesWithSuperValues(mm->environmentsAsComponents());
    _fillPartialValuesWithSuperValues(mm->materialsAsComponents());
  }
  else
    throw ArgumentException(A_FUNCINFO, String::format("Invalid level={0}", level));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits> void
ItemMaterialVariableBase<Traits>::
_fillPartialValuesWithSuperValues(MeshComponentList components)
{
  MatVarSpace s = this->space();
  ENUMERATE_COMPONENT (ic, components) {
    IMeshComponent* c = *ic;
    if (!c->hasSpace(s))
      continue;
    ENUMERATE_COMPONENTCELL (iccell, c) {
      ComponentCell c = (*iccell).superCell();
      setValue(iccell._varIndex(), value(c._varIndex()));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retrieves or constructs a variable.
 *
 * Retrieves or constructs a variable of type \a VariableTrueType whose
 * characteristics are given in \a v.
 */
template <typename VariableTrueType> IMeshMaterialVariable*
MeshMaterialVariableCommonStaticImpl<VariableTrueType>::
getReference(const MaterialVariableBuildInfo& v, MatVarSpace mvs)
{
  typedef typename VariableTrueType::PrivatePartType PrivatePartType;
  typedef typename VariableTrueType::VariableRefType VariableRefType;

  MeshHandle mesh_handle = v.meshHandle();
  if (mesh_handle.isNull())
    ARCANE_FATAL("No mesh handle for material variable");

  IMeshMaterialMng* mat_mng = v.materialMng();

  //TODO: check if lock is necessary
  if (!mat_mng)
    mat_mng = IMeshMaterialMng::getReference(mesh_handle, true);

  if (!mat_mng)
    ARCANE_FATAL("No IMaterialMng for mesh");

  if (!v.itemFamilyName().null())
    ARCANE_FATAL("Can not declare material variable for specific family");

  if (!v.itemGroupName().null())
    ARCANE_FATAL("Can not declare partial material variable");

  // Regarde si l'option qui force la création des variables matériaux
  // même si on est une variable milieu est active.

  if (VariableTrueType::TraitsType::dimension() == 0)
    if (mat_mng->isAllocateScalarEnvironmentVariableAsMaterial())
      mvs = MatVarSpace::MaterialAndEnvironment;

  VariableBuildInfo gvar_bi(v);

  {
    Mutex::ScopedLock sl(mat_mng->variableLock());

    //! Retrieves (or creates) the associated global variable.
    VariableInfo gvar_vi = VariableRefType::_internalVariableInfo(gvar_bi);
    IVariable* g_var = PrivatePartType::getReference(gvar_bi, gvar_vi);

    PrivatePartType* true_ptr = dynamic_cast<PrivatePartType*>(g_var);

    // TODO: check if we should also include the name of m_material_mng.
    // If we do, it poses a problem with the global variable
    // associated. If we don't, we can only have
    // a single IMeshMaterialMng.
    IMeshMaterialVariable* mat_var = mat_mng->checkVariable(g_var);
    VariableTrueType* true_mat_var = 0;
    // \a mat_var is not null if the associated variable already exists.
    if (mat_var)
      true_mat_var = dynamic_cast<VariableTrueType*>(mat_var);
    else {
      VariableRefType* var_ref = new VariableRefType(gvar_bi);
      MaterialVariableBuildInfo mvbi(mat_mng, v);
      true_mat_var = new VariableTrueType(mvbi, true_ptr, var_ref, mvs);
      mat_mng->_internalApi()->addVariable(true_mat_var);
      // The manager is already created, we can allocate all info
      if (!mat_mng->_internalApi()->variablesIndexer().empty())
        true_mat_var->buildFromManager(false);
    }
    ARCANE_CHECK_POINTER(true_mat_var);
    if (mvs != true_mat_var->space())
      ARCANE_FATAL("Incoherent space for variable '{0}' wanted_space={1} existing_space={2}.\n"
                   " Check all instances of the variables are of the same space\n"
                   " (i.e. all variables are MaterialVariable or EnvironmentVariable)",
                   true_mat_var->name(), (int)mvs, (int)true_mat_var->space());
    // Increments the reference counter.
    // It must be done here and not in addVariableRef() because in multithreading
    // it is possible to create multiple references to the same variable at the same
    // time and destroy one of the references before the addVariableRef() of the other
    // reference is called, which causes the destruction of true_mat_var.
    ++true_mat_var->m_p->m_nb_reference;
    return true_mat_var;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TrueType> MeshMaterialVariableFactoryRegisterer
MeshMaterialVariableCommonStaticImpl<TrueType>::
m_auto_registerer1(_autoCreate1, TrueType::BuilderType::_buildVarTypeInfo(MatVarSpace::Environment));

template <typename TrueType> MeshMaterialVariableFactoryRegisterer
MeshMaterialVariableCommonStaticImpl<TrueType>::
m_auto_registerer2(_autoCreate2, TrueType::BuilderType::_buildVarTypeInfo(MatVarSpace::MaterialAndEnvironment));

template <typename TrueType> IMeshMaterialVariable*
MeshMaterialVariableCommonStaticImpl<TrueType>::
_autoCreate1(const MaterialVariableBuildInfo& v)
{
  return getReference(v, MatVarSpace::Environment);
}

template <typename TrueType> IMeshMaterialVariable*
MeshMaterialVariableCommonStaticImpl<TrueType>::
_autoCreate2(const MaterialVariableBuildInfo& v)
{
  return getReference(v, MatVarSpace::MaterialAndEnvironment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
