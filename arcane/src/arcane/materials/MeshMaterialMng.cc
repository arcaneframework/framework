// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialMng.cc                                          (C) 2000-2026 */
/*                                                                           */
/* Material and mesh environment manager.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/MeshMaterialMng.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/AutoDestroyUserData.h"
#include "arcane/utils/IUserDataList.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/materials/IMeshMaterialVariableFactoryMng.h"
#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/MeshMaterialVariableRef.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MeshMaterialExchangeMng.h"
#include "arcane/materials/EnumeratorTracer.h"
#include "arcane/materials/MeshMaterialVariableFactoryRegisterer.h"
#include "arcane/materials/internal/AllEnvData.h"
#include "arcane/materials/internal/MeshMaterialModifierImpl.h"
#include "arcane/materials/internal/MeshMaterialSynchronizer.h"
#include "arcane/materials/internal/MeshMaterialVariableSynchronizer.h"
#include "arcane/materials/internal/ConstituentConnectivityList.h"
#include "arcane/materials/internal/AllCellToAllEnvCellContainer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file MaterialsGlobal.h
 *
 * Global declarations for materials.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO:
 * - Verify that only one instance of MeshModifier is created.
 * - For example, check in synchronizeMaterialsInCells()
 * that the mesh is not being modified.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

extern "C++" IMeshMaterialVariableFactoryMng*
arcaneCreateMeshMaterialVariableFactoryMng(IMeshMaterialMng* mm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  IMeshMaterialMng*
  arcaneCreateMeshMaterialMng(const MeshHandle& mesh_handle, const String& name)
  {
    MeshMaterialMng* mmm = new MeshMaterialMng(mesh_handle, name);
    //std::cout << "CREATE MESH_MATERIAL_MNG mesh_name=" << mesh_handle.meshName()
    //          << " ref=" << mesh_handle.reference() << " this=" << mmm << "\n";
    mmm->build();
    return mmm;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialMng::RunnerInfo::
RunnerInfo(Runner& runner)
: m_runner(runner)
, m_run_queue(makeQueue(m_runner))
, m_sequential_runner(Accelerator::eExecutionPolicy::Sequential)
, m_sequential_run_queue(makeQueue(m_sequential_runner))
, m_multi_thread_runner(Accelerator::eExecutionPolicy::Thread)
, m_multi_thread_run_queue(makeQueue(m_multi_thread_runner))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::RunnerInfo::
initializeAsyncPool(Int32 nb_queue)
{
  // If an accelerator policy is used, create asynchronous RunQueues
  // for independent operations. This will allow several to be executed
  // at the same time.
  bool is_accelerator = isAcceleratorPolicy(m_runner.executionPolicy());
  m_async_queue_pool.initialize(m_runner, nb_queue);
  if (is_accelerator)
    m_async_queue_pool.setAsync(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue MeshMaterialMng::RunnerInfo::
runQueue(Accelerator::eExecutionPolicy policy) const
{
  if (policy == Accelerator::eExecutionPolicy::None)
    return m_run_queue;
  if (policy == Accelerator::eExecutionPolicy::Sequential)
    return m_sequential_run_queue;
  if (policy == Accelerator::eExecutionPolicy::Thread)
    return m_multi_thread_run_queue;
  ARCANE_FATAL("Invalid value '{0}' for execution policy. Valid values are None, Sequential or Thread", policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialMng::
MeshMaterialMng(const MeshHandle& mesh_handle, const String& name)
// TODO: use the mesh's ITraceMng. Do it during init
: TraceAccessor(mesh_handle.traceMng())
, m_mesh_handle(mesh_handle)
, m_internal_api(std::make_unique<InternalApi>(this))
, m_variable_mng(mesh_handle.variableMng())
, m_name(name)
, m_indexed_selection_identity(MemoryUtils::getDefaultDataAllocator())
{
  m_all_env_data = std::make_unique<AllEnvData>(this);
  m_exchange_mng = std::make_unique<MeshMaterialExchangeMng>(this);
  m_variable_factory_mng = arcaneCreateMeshMaterialVariableFactoryMng(this);
  m_observer_pool = std::make_unique<ObserverPool>();
  m_observer_pool->addObserver(this, &MeshMaterialMng::_onMeshDestroyed, mesh_handle.onDestroyObservable());

  String s = platform::getEnvironmentVariable("ARCANE_ALLENVCELL_FOR_RUNCOMMAND");
  if (!s.null())
    m_is_use_accelerator_envcell_container = true;
  m_mms = new MeshMaterialSynchronizer(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialMng::
~MeshMaterialMng()
{
  //std::cout << "DESTROY MESH MATERIAL MNG this=" << this << '\n';
  _dumpStats();

  delete m_mms;
  delete m_variable_factory_mng;
  m_exchange_mng.reset();
  m_all_cells_env_only_synchronizer.reset();
  m_all_cells_mat_env_synchronizer.reset();
  m_all_env_data.reset();
  m_properties.reset();

  for (MeshMaterial* m : m_true_materials)
    delete m;
  m_true_materials.clear();

  for (MeshEnvironment* e : m_true_environments)
    delete e;
  m_true_environments.clear();

  for (IMeshBlock* b : m_true_blocks)
    delete b;

  for (MeshMaterialInfo* mmi : m_materials_info)
    delete mmi;

  for (MeshMaterialVariableIndexer* mvi : m_variables_indexer_to_destroy)
    delete mvi;

  m_modifier.reset();
  m_internal_api.reset();

  m_accelerator_envcell_container.reset();

  // Destroy the Runner at the end to ensure there are no more
  // references to it in other instances.
  m_runner_info.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
build()
{
  // Register variable factories
  {
    auto* x = MeshMaterialVariableFactoryRegisterer::firstRegisterer();
    while (x) {
      m_variable_factory_mng->registerFactory(x->createFactory());
      x = x->nextRegisterer();
    }
  }

  // Indicate whether the accelerator API is used for calculating
  // ConstituentItemVectorImpl entities
  {
    if (auto v = Convert::Type<Real>::tryParseFromEnvironment("ARCANE_MATERIALMNG_USE_ACCELERATOR_FOR_CONSTITUENTITEMVECTOR", true)) {
      m_is_use_accelerator_for_constituent_item_vector = (v.value() != 0);
    }
    // Do not activate the use of RunQueue for calculating
    // 'ComponentItemVector' if multi-threading is active. Currently,
    // using the same RunQueue is not multi-threaded (and therefore
    // ComponentItemVector cannot be created concurrently)
    if (TaskFactory::isActive())
      m_is_use_accelerator_for_constituent_item_vector = false;
    info() << "Use accelerator API for 'ConstituentItemVectorImpl' = " << m_is_use_accelerator_for_constituent_item_vector;
  }

  // Position the default runner
  {
    IAcceleratorMng* acc_mng = m_variable_mng->_internalApi()->acceleratorMng();
    Runner runner;
    if (acc_mng) {
      Runner* default_runner = acc_mng->defaultRunner();
      // Indicate whether the accelerator queue is activated
      bool use_accelerator_runner = true;
      if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_MATERIALMNG_USE_QUEUE", true))
        use_accelerator_runner = (v.value() != 0);
      if (use_accelerator_runner && default_runner)
        runner = *default_runner;
    }
    // If no runner is registered, use a sequential runner.
    if (!runner.isInitialized())
      runner.initialize(Accelerator::eExecutionPolicy::Sequential);
    m_runner_info = std::make_unique<RunnerInfo>(runner);
    Int32 nb_queue = isAcceleratorPolicy(runner.executionPolicy()) ? 8 : 1;
    info() << "Use runner '" << this->runner().executionPolicy() << "' for MeshMaterialMng name=" << name()
           << " async_queue_size=" << nb_queue;
    m_runner_info->initializeAsyncPool(nb_queue);

    // In release mode and if an accelerator is used, allocate by
    // default on the accelerator. This is important especially for
    // temporary arrays.
    // In 'check' mode, unified memory must be left because tests are done
    // on the CPU.
    RunQueue& q = runQueue();
    if (!arcaneIsCheck() && q.isAcceleratorPolicy())
      q.setMemoryRessource(eMemoryRessource::Device);
  }

  // Choice of optimizations.
  {
    int default_flags = 0;

    // Do not set these flags by default yet because it does not work
    // for all codes
    // default_flags = (int)eModificationFlags::GenericOptimize | (int)eModificationFlags::OptimizeMultiAddRemove;

    int opt_flag_value = 0;
    String env_name = "ARCANE_MATERIAL_MODIFICATION_FLAGS";
    String opt_flag_str = platform::getEnvironmentVariable(env_name);
    if (!opt_flag_str.null()) {
      if (builtInGetValue(opt_flag_value, opt_flag_str)) {
        pwarning() << "Invalid value '" << opt_flag_str
                   << " 'for environment variable '" << env_name
                   << "'";
        opt_flag_value = default_flags;
      }
    }
    else {
      opt_flag_value = default_flags;
    }
    m_modification_flags = opt_flag_value;
  }

  // Choice of synchronization implementation version
  {
    String env_name = "ARCANE_MATSYNCHRONIZE_VERSION";
    String env_value = platform::getEnvironmentVariable(env_name);
    info() << "ENV_VALUE=" << env_value;
    Integer version = m_synchronize_variable_version;
    if (!env_value.null()) {
      if (builtInGetValue(version, env_value)) {
        pwarning() << "Invalid value '" << env_value
                   << " 'for environment variable '" << env_name
                   << "'";
      }
      else
        m_synchronize_variable_version = version;
    }
    info() << "Set material variable synchronize version to "
           << "'" << m_synchronize_variable_version << "'";
  }

  // Choice of compression service
  {
    String env_name = "ARCANE_MATERIAL_DATA_COMPRESSOR_NAME";
    String env_value = platform::getEnvironmentVariable(env_name);
    if (!env_value.null()) {
      info() << "Use service '" << env_value << "' for material data compression";
      m_data_compressor_service_name = env_value;
    }
  }

  // Choice of additional capacity ratio
  {
    if (auto v = Convert::Type<Real>::tryParseFromEnvironment("ARCANE_MATERIALMNG_ADDITIONAL_CAPACITY_RATIO", true)) {
      if (v >= 0.0) {
        m_additional_capacity_ratio = v.value();
        info() << "Set additional capacity ratio to " << m_additional_capacity_ratio;
      }
    }
  }

  m_exchange_mng->build();
  // If enumerator traces on entities are active, activate those
  // on materials.
  // TODO: make this code thread-safe in case of using IParallelMng via threads
  // and call it only once.
  IItemEnumeratorTracer* item_tracer = IItemEnumeratorTracer::singleton();
  if (item_tracer) {
    info() << "Adding material enumerator tracing";
    EnumeratorTracer::_setSingleton(new EnumeratorTracer(traceMng(), item_tracer->perfCounterRef()));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_addVariableIndexer(MeshMaterialVariableIndexer* var_idx)
{
  var_idx->setIndex(m_variables_indexer.size());
  m_variables_indexer.add(var_idx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creation of a material.
 *
 * Creates a material named \a name, in environment \a env, with
 * info \a infos.
 */
MeshMaterial* MeshMaterialMng::
_createMaterial(MeshEnvironment* env, MeshMaterialInfo* infos, const String& name)
{
  _checkEndCreate();
  if (infos->materialMng() != this)
    ARCANE_FATAL("Invalid materialMng() for material info");
  if (env->materialMng() != this)
    ARCANE_FATAL("Invalid materialMng() for environment");
  Integer var_index = m_variables_indexer.size();
  Int16 mat_id = CheckedConvert::toInt16(m_materials.size());
  MeshMaterial* mat = new MeshMaterial(infos, env, name, mat_id);
  info() << "Create material name=" << name << "mat_id=" << mat_id << " var_index=" << var_index;
  mat->build();
  m_materials.add(mat);
  m_materials_as_components.add(mat);
  m_true_materials.add(mat);

  _addVariableIndexer(mat->variableIndexer());
  return mat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialInfo* MeshMaterialMng::
registerMaterialInfo(const String& name)
{
  _checkEndCreate();
  // Check that the material is not already registered.
  MeshMaterialInfo* old_mmi = _findMaterialInfo(name);
  if (old_mmi)
    ARCANE_FATAL("A material named '{0}' is already registered", name);

  MeshMaterialInfo* mmi = new MeshMaterialInfo(this, name);
  m_materials_info.add(mmi);
  return mmi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creation of an environment.
 *
 * The environment info is provided by the structure \a infos.
 * Along with the environment, all constituent materials are created.
 */
IMeshEnvironment* MeshMaterialMng::
createEnvironment(const MeshEnvironmentBuildInfo& infos)
{
  _checkEndCreate();
  Int16 env_index = CheckedConvert::toInt16(m_environments.size());
  // Check that an environment with the same name does not exist.
  const String& env_name = infos.name();
  MeshEnvironment* old_me = _findEnvironment(env_name);
  if (old_me)
    ARCANE_FATAL("An environment named '{0}' is already registered", env_name);

  info() << "Creating environment name=" << env_name << " index=" << env_index;
  // Create the environment
  MeshEnvironment* me = new MeshEnvironment(this, env_name, env_index);
  me->build();
  m_true_environments.add(me);
  m_environments.add(me);
  m_environments_as_components.add(me);

  // Create and add the materials
  Integer nb_mat = infos.materials().size();
  ConstArrayView<MeshEnvironmentBuildInfo::MatInfo> mat_build_infos = infos.materials();
  for (Integer i = 0; i < nb_mat; ++i) {
    const MeshEnvironmentBuildInfo::MatInfo& buildinfo = mat_build_infos[i];
    const String& mat_name = buildinfo.m_name;
    String new_mat_name = env_name + "_" + mat_name;
    MeshMaterialInfo* mat_info = _findMaterialInfo(mat_name);
    if (!mat_info) {
      ARCANE_FATAL("No material named '{0}' is defined", mat_name);
    }
    MeshMaterial* mm = _createMaterial(me, mat_info, new_mat_name);
    me->addMaterial(mm);
    mat_info->_addEnvironment(env_name);
  }
  // If the environment contains multiple materials, it must be allocated
  // with partial values. Otherwise, its partial values are those
  // of its unique material.
  {
    MeshMaterialVariableIndexer* var_idx = nullptr;
    if (nb_mat == 1) {
      var_idx = me->materials()[0]->_internalApi()->variableIndexer();
    }
    else {
      var_idx = new MeshMaterialVariableIndexer(traceMng(), me->name());
      _addVariableIndexer(var_idx);
      m_variables_indexer_to_destroy.add(var_idx);
    }
    me->setVariableIndexer(var_idx);
  }
  return me;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshBlock* MeshMaterialMng::
createBlock(const MeshBlockBuildInfo& infos)
{
  _checkEndCreate();

  Int32 block_index = m_blocks.size();
  // Checks that a block with the same name does not exist.
  const String& name = infos.name();
  const MeshBlock* old_mb = _findBlock(name);
  if (old_mb)
    ARCANE_FATAL("Un bloc de nom '{0}' est déjà enregistré", name);

  info() << "Creating block name=" << name << " index=" << block_index
         << " nb_env=" << infos.environments().size();
  Integer nb_env = infos.environments().size();
  for (Integer i = 0; i < nb_env; ++i)
    info() << " Adding environment name=" << infos.environments()[i]->name() << " to block";

  // Create the block
  MeshBlock* mb = new MeshBlock(this, block_index, infos);
  mb->build();
  m_true_blocks.add(mb);
  m_blocks.add(mb);

  return mb;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
addEnvironmentToBlock(IMeshBlock* block, IMeshEnvironment* env)
{
  MeshBlock* mb = ARCANE_CHECK_POINTER(dynamic_cast<MeshBlock*>(block));
  mb->addEnvironment(env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
removeEnvironmentToBlock(IMeshBlock* block, IMeshEnvironment* env)
{
  MeshBlock* mb = ARCANE_CHECK_POINTER(dynamic_cast<MeshBlock*>(block));
  mb->removeEnvironment(env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
endCreate(bool is_continue)
{
  if (m_is_end_create)
    return;

  _saveInfosInProperties();

  info() << "END CREATE MATERIAL_MNG is_continue=" << is_continue;

  m_modifier = std::make_unique<MeshMaterialModifierImpl>(this);
  m_modifier->initOptimizationFlags();

  m_all_env_data->endCreate(is_continue);

  auto synchronizer = mesh()->cellFamily()->allItemsSynchronizer();
  m_all_cells_mat_env_synchronizer = std::make_unique<MeshMaterialVariableSynchronizer>(this, synchronizer, MatVarSpace::MaterialAndEnvironment);
  m_all_cells_env_only_synchronizer = std::make_unique<MeshMaterialVariableSynchronizer>(this, synchronizer, MatVarSpace::Environment);

  // Determines the list of all components.
  {
    Integer nb_component = m_environments_as_components.size() + m_materials_as_components.size();
    m_components.reserve(nb_component);
    m_components.addRange(m_environments_as_components);
    m_components.addRange(m_materials_as_components);
  }

  // It is necessary to build and initialize the variables that were
  // created before this allocation.
  for (const auto& i : m_full_name_variable_map) {
    IMeshMaterialVariable* mv = i.second;
    info(4) << "BUILD FROM MANAGER name=" << mv->name() << " this=" << this;
    mv->buildFromManager(is_continue);
  }
  if (is_continue)
    _endUpdate();
  m_is_end_create = true;

  // Checks that the environments are valid.
  // NOTE: we cannot always call checkValid()
  // (especially at startup) because the entity groups exist,
  // but the associated material info are not
  // necessarily created yet.
  // (It will be necessary to check if this is due to compatible mode or not).
  for (IMeshEnvironment* env : m_environments) {
    env->checkValid();
  }

  // Now that everything is created, it is valid to register the mechanisms
  // of exchange.
  m_exchange_mng->registerFactory();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
setModificationFlags(int v)
{
  _checkEndCreate();
  m_modification_flags = v;
  info() << "Setting ModificationFlags to v=" << v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
setAllocateScalarEnvironmentVariableAsMaterial(bool v)
{
  _checkEndCreate();
  m_is_allocate_scalar_environment_variable_as_material = v;
  info() << "Setting AllocateScalarEnvironmentVariableAsMaterial to v=" << v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
setDataCompressorServiceName(const String& name)
{
  m_data_compressor_service_name = name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialModifierImpl* MeshMaterialMng::
_modifier()
{
  return m_modifier.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialInfo* MeshMaterialMng::
_findMaterialInfo(const String& name)
{
  for (MeshMaterialInfo* mmi : m_materials_info)
    if (mmi->name() == name)
      return mmi;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshEnvironment* MeshMaterialMng::
findEnvironment(const String& name, bool throw_exception)
{
  IMeshEnvironment* env = _findEnvironment(name);
  if (env)
    return env;
  if (throw_exception)
    ARCANE_FATAL("No environment named '{0}'", name);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshEnvironment* MeshMaterialMng::
_findEnvironment(const String& name)
{
  for (MeshEnvironment* env : m_true_environments)
    if (env->name() == name)
      return env;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshBlock* MeshMaterialMng::
findBlock(const String& name, bool throw_exception)
{
  IMeshBlock* block = _findBlock(name);
  if (block)
    return block;
  if (throw_exception)
    ARCANE_FATAL("No block named '{0}'", name);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshBlock* MeshMaterialMng::
_findBlock(const String& name)
{
  for (MeshBlock* b : m_true_blocks)
    if (b->name() == name)
      return b;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
forceRecompute()
{
  _endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Updates the structures following a modification of material or
 * environment cells.
 */
void MeshMaterialMng::
_endUpdate()
{
  m_all_env_data->forceRecompute(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Updates the variable references.
 *
 * This must be done when the number of elements per material or environment
 * changes because the arrays containing the associated variables may be
 * modified during the operation.
 */
void MeshMaterialMng::
syncVariablesReferences(bool check_resize)
{
  for (const auto& i : m_full_name_variable_map) {
    IMeshMaterialVariable* mv = i.second;
    info(4) << "SYNC REFERENCES FROM MANAGER name=" << mv->name();
    mv->_internalApi()->syncReferences(check_resize);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
visitVariables(IFunctorWithArgumentT<IMeshMaterialVariable*>* functor)
{
  if (!functor)
    return;
  for (const auto& i : m_full_name_variable_map) {
    IMeshMaterialVariable* mv = i.second;
    functor->executeFunctor(mv);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
checkValid()
{
  const IItemFamily* cell_family = mesh()->cellFamily();
  ItemGroup all_cells = cell_family->allItems();
  ConstArrayView<Int16> nb_env_per_cell = m_all_env_data->componentConnectivityList()->cellsNbEnvironment();
  ENUMERATE_ALLENVCELL (iallenvcell, view(all_cells.view().localIds())) {
    AllEnvCell all_env_cell = *iallenvcell;
    Integer cell_nb_env = all_env_cell.nbEnvironment();
    Cell cell = all_env_cell.globalCell();
    Int64 cell_uid = cell.uniqueId();
    if (all_env_cell.level() != LEVEL_ALLENVIRONMENT)
      ARCANE_FATAL("Bad level for all_env_item");

    if (all_env_cell.globalCell() != cell)
      ARCANE_FATAL("Bad corresponding globalCell() in all_env_item");
    if (cell_nb_env != nb_env_per_cell[cell.localId()])
      ARCANE_FATAL("Bad value for nb_env direct='{0}' var='{1}'",
                   cell_nb_env, nb_env_per_cell[cell.localId()]);
    for (Integer z = 0; z < cell_nb_env; ++z) {
      EnvCell ec = all_env_cell.cell(z);
      Integer cell_nb_mat = ec.nbMaterial();
      matimpl::ConstituentItemBase eii = ec.constituentItemBase();
      if (all_env_cell.constituentItemBase() != eii._superItemBase())
        ARCANE_FATAL("Bad corresponding allEnvItem() in env_item uid={0}", cell_uid);
      if (eii.globalItemBase() != cell)
        ARCANE_FATAL("Bad corresponding globalItem() in env_item");
      if (eii.level() != LEVEL_ENVIRONMENT)
        ARCANE_FATAL("Bad level '{0}' for in env_item", eii.level());
      // If the cell is not pure, the environment variable cannot be equivalent to
      // the global variable.
      if (cell_nb_env > 1 && ec._varIndex().arrayIndex() == 0)
        ARCANE_FATAL("Global index for a partial cell env_item={0}", ec);

      for (Integer k = 0; k < cell_nb_mat; ++k) {
        MatCell mc = ec.cell(k);
        matimpl::ConstituentItemBase mci = mc.constituentItemBase();
        if (eii != mci._superItemBase())
          ARCANE_FATAL("Bad corresponding env_item in mat_item k={0} mc={1}", k, mc);
        if (mci.globalItemBase() != cell)
          ARCANE_FATAL("Bad corresponding globalItem() in mat_item");
        if (mci.level() != LEVEL_MATERIAL)
          ARCANE_FATAL("Bad level '{0}' for in mat_item", mci.level());
        // If the cell is not pure, the material variable cannot be equivalent to
        // the global variable.
        if ((cell_nb_env > 1 || cell_nb_mat > 1) && mc._varIndex().arrayIndex() == 0) {
          ARCANE_FATAL("Global index for a partial cell matitem={0} name={1} nb_mat={2} nb_env={3}",
                       mc, mc.material()->name(), cell_nb_mat, cell_nb_env);
        }
      }
    }
  }

  for (IMeshEnvironment* env : m_environments) {
    env->checkValid();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialVariable* MeshMaterialMng::
findVariable(const String& name)
{
  IMeshMaterialVariable* v = _findVariableFullyQualified(name);
  if (v)
    return v;

  // Searches for the global variable named \a name
  // and if found, takes its full name for
  // the material variable.
  const IVariable* global_var = m_variable_mng->findMeshVariable(mesh(), name);
  if (global_var) {
    v = _findVariableFullyQualified(global_var->fullName());
    if (v)
      return v;
  }

  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialVariable* MeshMaterialMng::
_findVariableFullyQualified(const String& name)
{
  auto i = m_full_name_variable_map.find(name);
  if (i != m_full_name_variable_map.end())
    return i->second;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialVariable* MeshMaterialMng::
checkVariable(IVariable* global_var)
{
  auto i = m_var_to_mat_var_map.find(global_var);
  if (i != m_var_to_mat_var_map.end())
    return i->second;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
fillWithUsedVariables(Array<IMeshMaterialVariable*>& variables)
{
  variables.clear();

  // Uses the map based on variable names to ensure the same
  // traversal order regardless of sub-domains.
  for (const auto& i : m_full_name_variable_map) {
    IMeshMaterialVariable* ivar = i.second;
    if (ivar->globalVariable()->isUsed())
      variables.add(ivar);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_addVariable(IMeshMaterialVariable* var)
{
  //TODO: the lock m_variable_lock must be active.
  IVariable* gvar = var->globalVariable();
  info(4) << "MAT_ADD_VAR global_var=" << gvar << " var=" << var << " this=" << this;
  m_var_to_mat_var_map.insert(std::make_pair(gvar, var));
  m_full_name_variable_map.insert(std::make_pair(gvar->fullName(), var));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_removeVariable(IMeshMaterialVariable* var)
{
  //TODO: the lock m_variable_lock must be active.
  IVariable* gvar = var->globalVariable();
  info(4) << "MAT:Remove variable global_var=" << gvar << " var=" << var;
  m_var_to_mat_var_map.erase(gvar);
  m_full_name_variable_map.erase(gvar->fullName());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
dumpInfos(std::ostream& o)
{
  Integer nb_mat = m_materials.size();
  Integer nb_env = m_environments.size();
  Integer nb_var_idx = m_variables_indexer.size();
  o << "-- Infos sur les milieux et matériaux\n";
  o << "-- Nb Materiaux:            " << nb_mat << '\n';
  o << "-- Nb Milieux:              " << nb_env << '\n';
  o << "-- Nb Variables partielles: " << nb_var_idx << '\n';

  o << "-- Liste des matériaux\n";
  for (IMeshMaterial* mat : m_materials) {
    o << "--   Materiau name=" << mat->name() << '\n';
  }

  o << "-- Liste des milieux\n";
  for (IMeshEnvironment* me : m_environments) {
    ConstArrayView<IMeshMaterial*> env_materials = me->materials();
    const MeshMaterialVariableIndexer* env_var_idx = me->_internalApi()->variableIndexer();
    Integer nb_env_mat = env_materials.size();
    o << "--   Milieu name=" << me->name()
      << " nb_mat=" << nb_env_mat
      << " nb_cell=" << me->cells().size()
      << " var_idx = " << env_var_idx->index()
      << " ids=" << env_var_idx->matvarIndexes()
      << '\n';
    for (IMeshMaterial* mm : env_materials) {
      const MeshMaterialVariableIndexer* idx = mm->_internalApi()->variableIndexer();
      o << "--     Materiau\n";
      o << "--       name    = " << mm->name() << "\n";
      o << "--       nb_cell = " << mm->cells().size() << "\n";
      o << "--       var_idx = " << idx->index() << "\n";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge dumpInfos2() and dumpInfo().
void MeshMaterialMng::
dumpInfos2(std::ostream& o)
{
  const ConstituentConnectivityList& constituent_list = *m_all_env_data->componentConnectivityList();
  ConstArrayView<Int16> nb_env_per_cell = constituent_list.cellsNbEnvironment();
  Integer nb_mat = m_materials.size();
  Integer nb_env = m_environments.size();
  Integer nb_var_idx = m_variables_indexer.size();
  o << "-- Material and Environment infos: nb_env=" << nb_env
    << " nb_mat=" << nb_mat << " timestamp=" << m_timestamp
    << " nb_var_idx=" << nb_var_idx
    << "\n";
  Integer nb_cell = mesh()->allCells().size();
  if (nb_cell != 0) {
    Integer nb_pure_env = 0;
    ENUMERATE_CELL (icell, mesh()->allCells()) {
      if (nb_env_per_cell[icell.localId()] <= 1)
        ++nb_pure_env;
    }
    o << " nb_cell=" << nb_cell << " nb_pure_env=" << nb_pure_env
      << " nb_partial=" << (nb_cell - nb_pure_env)
      << " percent=" << (100 * nb_pure_env) / nb_cell
      << "\n";
  }

  o << "-- Liste des milieux\n";
  for (MeshEnvironment* me : m_true_environments) {
    ConstArrayView<IMeshMaterial*> env_materials = me->materials();
    const MeshMaterialVariableIndexer* env_var_idx = me->variableIndexer();
    const Int16 env_id = me->componentId();
    Integer nb_env_mat = env_materials.size();
    Integer nb_env_cell = me->cells().size();
    Integer nb_pure_mat = 0;
    if (nb_env_mat > 1) {
      ENUMERATE_CELL (icell, me->cells()) {
        if (constituent_list.cellNbMaterial(icell, env_id) <= 1)
          ++nb_pure_mat;
      }
    }
    else
      nb_pure_mat = nb_env_cell;
    o << "--   Env name=" << me->name()
      << " nb_mat=" << nb_env_mat
      << " var_idx=" << env_var_idx->index()
      << " nb_cell=" << nb_env_cell
      << " nb_pure_mat=" << nb_pure_mat;
    if (nb_env_cell != 0)
      o << " percent=" << (nb_pure_mat * 100) / nb_env_cell;
    o << '\n';
    for (Integer j = 0; j < nb_env_mat; ++j) {
      IMeshMaterial* mm = env_materials[j];
      const MeshMaterialVariableIndexer* idx = mm->_internalApi()->variableIndexer();
      o << "--     Mat name=" << mm->name()
        << " nb_cell=" << mm->cells().size()
        << " var_idx=" << idx->index()
        << "\n";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshMaterialMng::
synchronizeMaterialsInCells()
{
  return m_mms->synchronizeMaterialsInCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
checkMaterialsInCells(Integer max_print)
{
  m_mms->checkMaterialsInCells(max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
dumpCellInfos(Cell cell, std::ostream& o)
{
  CellToAllEnvCellConverter all_env_cell_converter(this);
  AllEnvCell all_env_cell = all_env_cell_converter[cell];
  Cell global_cell = all_env_cell.globalCell();
  o << "Cell uid=" << ItemPrinter(global_cell) << '\n';
  ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
    o << "ENV name=" << (*ienvcell).environment()->name()
      << " component_idx=" << ComponentItemLocalId(ienvcell) << '\n';
    ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
      o << "MAT name=" << (*imatcell).material()->name()
        << " component_idx=" << ComponentItemLocalId(imatcell) << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellToAllEnvCellConverter MeshMaterialMng::
cellToAllEnvCellConverter()
{
  return CellToAllEnvCellConverter(componentItemSharedInfo(LEVEL_ALLENVIRONMENT));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_checkEndCreate()
{
  if (m_is_end_create)
    ARCANE_FATAL("Invalid method call because endCreate() has already been called");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvCellVectorView MeshMaterialMng::
_view(SmallSpan<const Int32> local_ids)
{
  return AllEnvCellVectorView(local_ids.constSmallView(), componentItemSharedInfo(LEVEL_ALLENVIRONMENT));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialMngFactory
: public IMeshMaterialMng::IFactory
{
 public:

  MeshMaterialMngFactory()
  {
    IMeshMaterialMng::_internalSetFactory(this);
  }
  ~MeshMaterialMngFactory()
  {
    IMeshMaterialMng::_internalSetFactory(nullptr);
  }

 public:

  Ref<IMeshMaterialMng> getTrueReference(const MeshHandle& mesh_handle, bool is_create) override;

 public:

  static MeshMaterialMngFactory m_mesh_material_mng_factory;
};

MeshMaterialMngFactory MeshMaterialMngFactory::m_mesh_material_mng_factory{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMeshMaterialMng> MeshMaterialMngFactory::
getTrueReference(const MeshHandle& mesh_handle, bool is_create)
{
  //TODO: implement lock for multi-threading
  typedef AutoDestroyUserData<Ref<IMeshMaterialMng>> UserDataType;

  const char* name = "MeshMaterialMng_StdMat";
  IUserDataList* udlist = mesh_handle.meshUserDataList();

  IUserData* ud = udlist->data(name, true);
  if (!ud) {
    if (!is_create)
      return {};
    IMeshMaterialMng* mm = arcaneCreateMeshMaterialMng(mesh_handle, "StdMat");
    Ref<IMeshMaterialMng> mm_ref = makeRef(mm);
    udlist->setData(name, new UserDataType(new Ref<IMeshMaterialMng>(mm_ref)));
    return mm_ref;
  }
  auto adud = dynamic_cast<UserDataType*>(ud);
  if (!adud)
    ARCANE_FATAL("Can not cast to IMeshMaterialMng*");
  return *(adud->data());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshMaterialMng::
isInMeshMaterialExchange() const
{
  return m_exchange_mng->isInMeshMaterialExchange();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_checkCreateProperties()
{
  if (m_properties)
    return;
  m_properties = std::make_unique<Properties>(*(mesh()->properties()), String("MeshMaterialMng_") + name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
  const Int32 SERIALIZE_VERSION = 1;
}
void MeshMaterialMng::
_saveInfosInProperties()
{
  _checkCreateProperties();

  // Save the version number to ensure compatibility during recovery
  m_properties->set("Version", SERIALIZE_VERSION);

  // Save the necessary information in the properties to recreate the
  // materials and environments.
  UniqueArray<String> material_info_names;
  for (MeshMaterialInfo* mat_info : m_materials_info) {
    material_info_names.add(mat_info->name());
  }
  m_properties->set("MaterialInfoNames", material_info_names);

  UniqueArray<String> env_names;
  UniqueArray<Int32> env_nb_mat;
  UniqueArray<String> env_mat_names;
  ENUMERATE_ENV (ienv, this) {
    IMeshEnvironment* env = *ienv;
    env_names.add(env->name());
    info(5) << "SAVE ENV_NAME name=" << env->name() << " nb_mat=" << env->nbMaterial();
    env_nb_mat.add(env->nbMaterial());
    ENUMERATE_MAT (imat, env) {
      const String& name = (*imat)->infos()->name();
      info(5) << "SAVE MAT_NAME name=" << name;
      env_mat_names.add(name);
    }
  }
  m_properties->set("EnvNames", env_names);
  m_properties->set("EnvNbMat", env_nb_mat);
  m_properties->set("EnvMatNames", env_mat_names);

  // Save the necessary information for the blocks.
  // For each block, its name and the name of the corresponding cell group.
  UniqueArray<String> block_names;
  UniqueArray<String> block_cell_group_names;
  UniqueArray<Int32> block_nb_env;
  UniqueArray<String> block_env_names;
  for (IMeshBlock* block : m_blocks) {
    block_names.add(block->name());
    block_cell_group_names.add(block->cells().name());
    block_nb_env.add(block->nbEnvironment());
    ENUMERATE_ENV (ienv, block) {
      const String& name = (*ienv)->name();
      info(5) << "SAVE BLOCK ENV_NAME name=" << name;
      block_env_names.add(name);
    }
  }
  m_properties->set("BlockNames", block_names);
  m_properties->set("BlockCellGroupNames", block_cell_group_names);
  m_properties->set("BlockNbEnv", block_nb_env);
  m_properties->set("BlockEnvNames", block_env_names);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
recreateFromDump()
{
  if (m_is_end_create)
    ARCANE_FATAL("Can not recreate a created instance");

  _checkCreateProperties();

  info() << "Creating material infos from dump";

  // Save the version number to ensure compatibility during recovery
  Int32 v = m_properties->getInt32("Version");
  if (v != SERIALIZE_VERSION)
    ARCANE_FATAL("Bad serializer version: trying to read from incompatible checkpoint v={0} expected={1}",
                 v, SERIALIZE_VERSION);

  UniqueArray<String> material_info_names;
  m_properties->get("MaterialInfoNames", material_info_names);
  for (const String& mat_name : material_info_names)
    this->registerMaterialInfo(mat_name);

  UniqueArray<String> env_names;
  UniqueArray<Int32> env_nb_mat;
  UniqueArray<String> env_mat_names;
  m_properties->get("EnvNames", env_names);
  m_properties->get("EnvNbMat", env_nb_mat);
  m_properties->get("EnvMatNames", env_mat_names);

  Integer mat_index = 0;
  for (Integer ienv = 0, nenv = env_names.size(); ienv < nenv; ++ienv) {
    Materials::MeshEnvironmentBuildInfo env_build(env_names[ienv]);
    Integer nb_mat = env_nb_mat[ienv];
    for (Integer imat = 0; imat < nb_mat; ++imat) {
      env_build.addMaterial(env_mat_names[mat_index]);
      ++mat_index;
    }
    this->createEnvironment(env_build);
  }

  // Recreate the blocks.
  // For each block, its name and the name of the corresponding cell group.
  UniqueArray<String> block_names;
  UniqueArray<String> block_cell_group_names;
  UniqueArray<String> block_env_names;
  UniqueArray<Int32> block_nb_env;
  m_properties->get("BlockNames", block_names);
  m_properties->get("BlockCellGroupNames", block_cell_group_names);
  m_properties->get("BlockNbEnv", block_nb_env);
  m_properties->get("BlockEnvNames", block_env_names);
  const IItemFamily* cell_family = mesh()->cellFamily();
  Integer block_env_index = 0;
  for (Integer i = 0, n = block_names.size(); i < n; ++i) {
    String name = block_names[i];
    String cell_group_name = block_cell_group_names[i];
    CellGroup cells = cell_family->findGroup(cell_group_name);
    if (cells.null())
      ARCANE_FATAL("Can not find cell group '{0}' for block creation",
                   cell_group_name);
    MeshBlockBuildInfo mbbi(name, cells);
    if (!block_nb_env.empty()) {
      Integer nb_env = block_nb_env[i];
      for (Integer ienv = 0; ienv < nb_env; ++ienv) {
        const String& name2 = block_env_names[block_env_index];
        ++block_env_index;
        IMeshEnvironment* env = findEnvironment(name2, false);
        if (!env)
          ARCANE_FATAL("Invalid environment name '{0}' for recreating blocks", name2);
        mbbi.addEnvironment(env);
      }
    }
    this->createBlock(mbbi);
  }

  endCreate(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_onMeshDestroyed()
{
  // This instance must be destroyed here because it requires IItemFamily
  // in its destructor, and it is possible that the family no longer exists
  // if the IMeshMaterialMng destructor is called after the mesh destruction
  // (which can happen in C# for example).
  m_exchange_mng.reset();

  _unregisterAllVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_unregisterAllVariables()
{
  // Copy all references into an array.
  // This must be done before calling unregisterVariable()
  // because the latter modifies the linked list of references
  UniqueArray<MeshMaterialVariableRef*> m_all_refs;

  for (const auto& i : m_full_name_variable_map) {
    const IMeshMaterialVariable* var = i.second;

    for (MeshMaterialVariableRef::Enumerator iref(var); iref.hasNext(); ++iref) {
      MeshMaterialVariableRef* ref = *iref;
      m_all_refs.add(ref);
    }
  }

  for (MeshMaterialVariableRef* ref : m_all_refs)
    ref->unregisterVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemSharedInfo* MeshMaterialMng::
componentItemSharedInfo(Int32 level) const
{
  ComponentItemInternalData* data = m_all_env_data->componentItemInternalData();
  ComponentItemSharedInfo* shared_info = nullptr;
  if (level == LEVEL_MATERIAL)
    shared_info = data->matSharedInfo();
  else if (level == LEVEL_ENVIRONMENT)
    shared_info = data->envSharedInfo();
  else if (level == LEVEL_ALLENVIRONMENT)
    shared_info = data->allEnvSharedInfo();
  else
    ARCANE_FATAL("Bad internal type of component");

  return shared_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_dumpStats()
{
  IEnumeratorTracer* tracer = IEnumeratorTracer::singleton();
  if (tracer)
    tracer->dumpStats();

  if (m_modifier)
    m_modifier->dumpStats();

  for (IMeshEnvironment* env : m_environments) {
    // Do not display statistics if the environment has only one material
    // because it uses the same indexer as the material and the statistics
    // for it will be displayed when iterating over the materials.
    if (env->nbMaterial() > 1)
      env->_internalApi()->variableIndexer()->dumpStats();
  }
  for (IMeshMaterial* mat : m_materials) {
    mat->_internalApi()->variableIndexer()->dumpStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
createAllCellToAllEnvCell()
{
  if (!m_accelerator_envcell_container) {
    m_accelerator_envcell_container = std::make_unique<AllCellToAllEnvCellContainer>(this);
    m_accelerator_envcell_container->initialize();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SmallSpan<const Int32> MeshMaterialMng::
identitySelectionView()
{
  // NOTE: this array could perhaps be managed directly
  // by the family if there is interest in using it in other contexts
  Int32 max_local_id = m_mesh_handle.mesh()->cellFamily()->maxLocalId();
  {
    std::scoped_lock sl(m_indexed_selection_identity_mutex);
    Int32 size = m_indexed_selection_identity.size();
    if (max_local_id > size) {
      m_indexed_selection_identity.resize(max_local_id);
      for (Int32 i = size; i < max_local_id; ++i)
        m_indexed_selection_identity[i] = i;
    }
    return m_indexed_selection_identity.constView();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
