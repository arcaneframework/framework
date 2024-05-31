// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialMng.h                                           (C) 2000-2024 */
/*                                                                           */
/* Implémentation de la modification des matériaux et milieux.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALMNG_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Mutex.h"

#include "arcane/core/MeshHandle.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueuePool.h"

#include "arcane/materials/MeshBlock.h"
#include "arcane/materials/AllCellToAllEnvCellConverter.h"
#include "arcane/materials/internal/MeshMaterial.h"
#include "arcane/materials/internal/MeshEnvironment.h"

#include <map>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariableMng;
class Properties;
class ObserverPool;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'un gestion des matériaux.
 */
class MeshMaterialMng
: public TraceAccessor
, public IMeshMaterialMng
{
 public:

  friend class MeshMaterialBackup;

 private:

  //! Informations sur la file d'exécution utilisée
  class RunnerInfo
  {
   public:

    explicit RunnerInfo(Runner& runner);

   public:

    void initializeAsyncPool(Int32 nb_queue);

   public:

    Runner m_runner;
    RunQueue m_run_queue;
    Accelerator::RunQueuePool m_async_queue_pool;
  };

  class InternalApi
  : public IMeshMaterialMngInternal
  {
   public:

    explicit InternalApi(MeshMaterialMng* mm)
    : m_material_mng(mm)
    {}

   public:

    AllCellToAllEnvCell* getAllCellToAllEnvCell() const override
    {
      return m_material_mng->getAllCellToAllEnvCell();
    }
    void createAllCellToAllEnvCell(IMemoryAllocator* alloc) override
    {
      return m_material_mng->createAllCellToAllEnvCell(alloc);
    }
    ConstArrayView<MeshMaterialVariableIndexer*> variablesIndexer() override
    {
      return m_material_mng->_variablesIndexer();
    }
    void addVariable(IMeshMaterialVariable* var) override
    {
      return m_material_mng->_addVariable(var);
    }
    void removeVariable(IMeshMaterialVariable* var) override
    {
      return m_material_mng->_removeVariable(var);
    }
    MeshMaterialModifierImpl* modifier() override
    {
      return m_material_mng->_modifier();
    }
    IMeshMaterialVariableSynchronizer* allCellsMatEnvSynchronizer() override
    {
      return m_material_mng->_allCellsMatEnvSynchronizer();
    }
    IMeshMaterialVariableSynchronizer* allCellsEnvOnlySynchronizer() override
    {
      return m_material_mng->_allCellsEnvOnlySynchronizer();
    }
    ComponentItemSharedInfo* componentItemSharedInfo(Int32 level) const override
    {
      return m_material_mng->componentItemSharedInfo(level);
    }
    RunQueue& runQueue() const override
    {
      return m_material_mng->runQueue();
    }
    Accelerator::RunQueuePool& asyncRunQueuePool() const override
    {
      return m_material_mng->asyncRunQueuePool();
    }
    Real additionalCapacityRatio() const override
    {
      return m_material_mng->additionalCapacityRatio();
    }

   private:

    MeshMaterialMng* m_material_mng = nullptr;
  };

 public:

  MeshMaterialMng(const MeshHandle& mesh_handle,const String& name);
  ~MeshMaterialMng() override;

 public:

  void build();

 public:

  IMesh* mesh() override { return m_mesh_handle.mesh(); }
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }

 public:

  MeshMaterialInfo* registerMaterialInfo(const String& name) override;
  IMeshEnvironment* createEnvironment(const MeshEnvironmentBuildInfo& infos) override;
  IMeshBlock* createBlock(const MeshBlockBuildInfo& infos) override;
  void addEnvironmentToBlock(IMeshBlock* block,IMeshEnvironment* env) override;
  void removeEnvironmentToBlock(IMeshBlock* block,IMeshEnvironment* env) override;

  void endCreate(bool is_continue) override;

  void setDataInitialisationWithZero(bool v) override { m_is_data_initialisation_with_zero = v; }
  bool isDataInitialisationWithZero() const override { return m_is_data_initialisation_with_zero; }

  void setKeepValuesAfterChange(bool v) override { m_keep_values_after_change = v; }
  bool isKeepValuesAfterChange() const override { return m_keep_values_after_change; }

  void setMeshModificationNotified(bool v) override { m_is_mesh_modification_notified = v; }
  bool isMeshModificationNotified() const override { return m_is_mesh_modification_notified; }

  void setModificationFlags(int v) override;
  int modificationFlags() const override { return m_modification_flags; }

  void setAllocateScalarEnvironmentVariableAsMaterial(bool v) override;
  bool isAllocateScalarEnvironmentVariableAsMaterial() const override
  {
    return m_is_allocate_scalar_environment_variable_as_material;
  }

  void setDataCompressorServiceName(const String& name) override;
  String dataCompressorServiceName() const override { return m_data_compressor_service_name; }

  String name() const override { return m_name; }
  ConstArrayView<IMeshMaterial*> materials() const override { return m_materials; }
  ConstArrayView<IMeshComponent*> materialsAsComponents() const override { return m_materials_as_components; }
  ConstArrayView<IMeshEnvironment*> environments() const override { return m_environments; }
  ConstArrayView<IMeshComponent*> environmentsAsComponents() const override { return m_environments_as_components; }
  ConstArrayView<IMeshComponent*> components() const override { return m_components; }
  ConstArrayView<IMeshBlock*> blocks() const override { return m_blocks; }

  IMeshEnvironment* findEnvironment(const String& name,bool throw_exception=true) override;
  IMeshBlock* findBlock(const String& name,bool throw_exception=true) override;

  void fillWithUsedVariables(Array<IMeshMaterialVariable*>& variables) override;

  IMeshMaterialVariable* findVariable(const String& name) override;
  IMeshMaterialVariable* checkVariable(IVariable* global_var) override;

  void dumpInfos(std::ostream& o) override;
  void dumpCellInfos(Cell cell,std::ostream& o) override;

  void checkValid() override;

  void forceRecompute() override;

  Mutex* variableLock() override
  {
    return &m_variable_lock;
  }

  bool synchronizeMaterialsInCells() override;
  void checkMaterialsInCells(Integer max_print) override;

  Int64 timestamp() const override { return m_timestamp; }

  ConstArrayView<MeshEnvironment*> trueEnvironments() const { return m_true_environments; }
  ConstArrayView<MeshMaterial*> trueMaterials() const { return m_true_materials; }

 public:

  AllEnvCellVectorView view(Int32ConstArrayView local_ids);

  AllEnvCellVectorView view(const CellGroup& cells) override
  {
    return this->view(cells.view().localIds());
  }

  AllEnvCellVectorView view(CellVectorView cells) override
  {
    return this->view(cells.localIds());
  }

  CellToAllEnvCellConverter cellToAllEnvCellConverter() override;

  void recreateFromDump() override;

  void visitVariables(IFunctorWithArgumentT<IMeshMaterialVariable*>* functor) override;

  void setSynchronizeVariableVersion(Integer version) override
  {
    m_synchronize_variable_version = version;
  }

  Integer synchronizeVariableVersion() const override
  {
    return m_synchronize_variable_version;
  }

  bool isInMeshMaterialExchange() const override;

  IMeshMaterialVariableFactoryMng* variableFactoryMng() const override
  {
    return m_variable_factory_mng;
  }

  void setUseMaterialValueWhenRemovingPartialValue(bool v) override
  {
    m_is_use_material_value_when_removing_partial_value = v;
  }
  bool isUseMaterialValueWhenRemovingPartialValue() const override
  {
    return m_is_use_material_value_when_removing_partial_value;
  }

 public:

  AllEnvData* allEnvData() { return m_all_env_data.get(); }
  ComponentItemSharedInfo* componentItemSharedInfo(Int32 level) const;
  void syncVariablesReferences(bool check_resize);

  void incrementTimestamp() { ++m_timestamp; }
  void dumpInfos2(std::ostream& o);

  const MeshHandle& meshHandle() const { return m_mesh_handle; }

  void enableCellToAllEnvCellForRunCommand(bool is_enable, bool force_create=false) override
  {
    m_is_allcell_2_allenvcell = is_enable;
    if (force_create)
      createAllCellToAllEnvCell(platform::getDefaultDataAllocator());
  }
  bool isCellToAllEnvCellForRunCommand() const override { return m_is_allcell_2_allenvcell; }

  IMeshMaterialMngInternal* _internalApi() const override { return m_internal_api.get(); }

 public:

  //@{ Implémentation de IMeshMaterialMngInternal
  Runner& runner() const { return m_runner_info->m_runner; }
  RunQueue& runQueue() const { return m_runner_info->m_run_queue; }
  Accelerator::RunQueuePool& asyncRunQueuePool() const { return m_runner_info->m_async_queue_pool; }
  Real additionalCapacityRatio() const { return 0.05; }
  //@}

 private:

  AllCellToAllEnvCell* getAllCellToAllEnvCell() const { return m_allcell_2_allenvcell; }
  void createAllCellToAllEnvCell(IMemoryAllocator* alloc)
  {
    if (!m_allcell_2_allenvcell)
      m_allcell_2_allenvcell = AllCellToAllEnvCell::create(this, alloc);
  }

 private:

  //! Type de la liste des variables par nom complet
  using FullNameVariableMap= std::map<String,IMeshMaterialVariable*>;
  //! Paire de la liste des variables par nom complet
  using FullNameVariablePair = FullNameVariableMap::value_type;

  using VariableToMaterialVariableMap = std::map<IVariable*,IMeshMaterialVariable*>;
  using VariableToMaterialVariablePair = VariableToMaterialVariableMap::value_type;

 private:

  MeshHandle m_mesh_handle;
  std::unique_ptr<InternalApi> m_internal_api;
  IVariableMng* m_variable_mng = nullptr;
  String m_name;
  bool m_is_end_create = false;
  bool m_is_verbose = false;
  bool m_keep_values_after_change = true;
  bool m_is_data_initialisation_with_zero = false;
  bool m_is_mesh_modification_notified = false;
  bool m_is_allocate_scalar_environment_variable_as_material = false;
  bool m_is_use_material_value_when_removing_partial_value = false;
  int m_modification_flags = 0;

  Mutex m_variable_lock;

  std::unique_ptr<MeshMaterialModifierImpl> m_modifier;
  UniqueArray<MeshMaterialInfo*> m_materials_info;
  UniqueArray<IMeshMaterial*> m_materials;
  UniqueArray<IMeshComponent*> m_materials_as_components;
  UniqueArray<MeshMaterial*> m_true_materials;
  UniqueArray<IMeshEnvironment*> m_environments;
  UniqueArray<IMeshComponent*> m_environments_as_components;
  UniqueArray<IMeshComponent*> m_components;
  UniqueArray<MeshEnvironment*> m_true_environments;
  UniqueArray<IMeshBlock*> m_blocks;
  UniqueArray<MeshBlock*> m_true_blocks;
  UniqueArray<MeshMaterialVariableIndexer*> m_variables_indexer;
  UniqueArray<MeshMaterialVariableIndexer*> m_variables_indexer_to_destroy;

  FullNameVariableMap m_full_name_variable_map;
  VariableToMaterialVariableMap m_var_to_mat_var_map;

  std::unique_ptr<Properties> m_properties;
  std::unique_ptr<AllEnvData> m_all_env_data;
  Int64 m_timestamp = 0; //!< Compteur du nombre de modifications des matériaux.
  std::unique_ptr<IMeshMaterialVariableSynchronizer> m_all_cells_mat_env_synchronizer;
  std::unique_ptr<IMeshMaterialVariableSynchronizer> m_all_cells_env_only_synchronizer;
  Integer m_synchronize_variable_version = 1;
  std::unique_ptr<MeshMaterialExchangeMng> m_exchange_mng;
  IMeshMaterialVariableFactoryMng* m_variable_factory_mng = nullptr;
  std::unique_ptr<ObserverPool> m_observer_pool;
  String m_data_compressor_service_name;

  AllCellToAllEnvCell* m_allcell_2_allenvcell = nullptr;
  bool m_is_allcell_2_allenvcell = false;

  std::unique_ptr<RunnerInfo> m_runner_info;

 private:

  void _endUpdate();
  IMeshMaterialVariable* _findVariableFullyQualified(const String& name);
  MeshMaterialInfo* _findMaterialInfo(const String& name);
  MeshEnvironment* _findEnvironment(const String& name);
  MeshBlock* _findBlock(const String& name);
  MeshMaterial* _createMaterial(MeshEnvironment* env,MeshMaterialInfo* infos,const String& name);
  void _addVariableIndexer(MeshMaterialVariableIndexer* var_idx);
  void _checkEndCreate();
  void _addVariableUnlocked(IMeshMaterialVariable* var);
  void _saveInfosInProperties();
  void _checkCreateProperties();
  void _onMeshDestroyed();
  void _unregisterAllVariables();
  void _addVariable(IMeshMaterialVariable* var);
  void _removeVariable(IMeshMaterialVariable* var);
  MeshMaterialModifierImpl* _modifier();
  ConstArrayView<MeshMaterialVariableIndexer*> _variablesIndexer()
  {
    return m_variables_indexer;
  }
  IMeshMaterialVariableSynchronizer* _allCellsMatEnvSynchronizer() override
  {
    return m_all_cells_mat_env_synchronizer.get();
  }
  IMeshMaterialVariableSynchronizer* _allCellsEnvOnlySynchronizer() override
  {
    return m_all_cells_env_only_synchronizer.get();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
