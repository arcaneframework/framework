// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialMng.h                                           (C) 2000-2023 */
/*                                                                           */
/* Implémentation de la modification des matériaux et milieux.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALMNG_H
#define ARCANE_MATERIALS_MESHMATERIALMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Mutex.h"

#include "arcane/core/MeshHandle.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MeshMaterial.h"
#include "arcane/materials/MeshEnvironment.h"
#include "arcane/materials/MeshBlock.h"
#include "arcane/materials/MatItemEnumerator.h"

#include <map>
#include <memory>

#include "arcane/materials/AllCellToAllEnvCellConverter.h"

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
class MeshMaterialModifierImpl;
class MeshMaterialBackup;
class AllEnvData;
class MeshMaterialExchangeMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'un gestion des matériaux.
 */
class MeshMaterialMng
: public TraceAccessor
, public IMeshMaterialMng
{
 public:
  
  friend class MeshMaterialBackup;

 public:

  MeshMaterialMng(const MeshHandle& mesh_handle,const String& name);
  virtual ~MeshMaterialMng() override;

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

  const String& name() const override { return m_name; }
  ConstArrayView<IMeshMaterial*> materials() const override { return m_materials; }
  ConstArrayView<IMeshComponent*> materialsAsComponents() const override { return m_materials_as_components; }
  ConstArrayView<IMeshEnvironment*> environments() const override { return m_environments; }
  ConstArrayView<IMeshComponent*> environmentsAsComponents() const override { return m_environments_as_components; }
  ConstArrayView<IMeshComponent*> components() const override { return m_components; }
  ConstArrayView<IMeshBlock*> blocks() const override { return m_blocks; }

  IMeshEnvironment* findEnvironment(const String& name,bool throw_exception=true) override;
  IMeshBlock* findBlock(const String& name,bool throw_exception=true) override;
  ConstArrayView<MeshMaterialVariableIndexer*> variablesIndexer() override
  {
    return m_variables_indexer;
  }
  IMeshMaterialModifierImpl* modifier() override;

  void fillWithUsedVariables(Array<IMeshMaterialVariable*>& variables) override;
  void addVariable(IMeshMaterialVariable* var) override;
  void removeVariable(IMeshMaterialVariable* var) override;

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

  void synchronizeMaterialsInCells() override;
  void checkMaterialsInCells(Integer max_print) override;

  Int64 timestamp() const override { return m_timestamp; }

  IMeshMaterialVariableSynchronizer* _allCellsMatEnvSynchronizer() override
  {
    return m_all_cells_mat_env_synchronizer;
  }
  IMeshMaterialVariableSynchronizer* _allCellsEnvOnlySynchronizer() override
  {
    return m_all_cells_env_only_synchronizer;
  }

  ConstArrayView<MeshEnvironment*> trueEnvironments() const { return m_true_environments; }

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

 public:

  AllEnvData* allEnvData() { return m_all_env_data; }
  void syncVariablesReferences();

  void incrementTimestamp() { ++m_timestamp; }
  void dumpInfos2(std::ostream& o);

  const MeshHandle& meshHandle() const { return m_mesh_handle; }

  //!\internal
  // TODO: POUR TESTER, A BLINDER SI ON GARDE
  AllCell2AllEnvCell* getAllCell2AllEnvCell() const { return m_allcell_2_allenvcell; }
  void createAllCell2AllEnvCell(IMemoryAllocator* alloc) { m_allcell_2_allenvcell = AllCell2AllEnvCell::create(this, alloc); }

 private:

  //! Type de la liste des variables par nom complet
  typedef std::map<String,IMeshMaterialVariable*> FullNameVariableMap;
  //! Paire de la liste des variables par nom complet
  typedef FullNameVariableMap::value_type FullNameVariablePair;

  typedef std::map<IVariable*,IMeshMaterialVariable*> VariableToMaterialVariableMap;
  typedef VariableToMaterialVariableMap::value_type VariableToMaterialVariablePair;

 private:

  MeshHandle m_mesh_handle;
  IVariableMng* m_variable_mng = nullptr;
  String m_name;
  bool m_is_end_create;
  bool m_is_verbose;
  bool m_keep_values_after_change;
  bool m_is_data_initialisation_with_zero;
  bool m_is_mesh_modification_notified;
  bool m_is_allocate_scalar_environment_variable_as_material;
  int m_modification_flags = 0;

  Mutex m_variable_lock;

  MeshMaterialModifierImpl* m_modifier = nullptr;
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

  Properties* m_properties = nullptr;
  AllEnvData* m_all_env_data = nullptr;
  Int64 m_timestamp; //!< Compteur du nombre de modifications des matériaux.
  IMeshMaterialVariableSynchronizer* m_all_cells_mat_env_synchronizer;
  IMeshMaterialVariableSynchronizer* m_all_cells_env_only_synchronizer;
  Integer m_synchronize_variable_version;
  MeshMaterialExchangeMng* m_exchange_mng = nullptr;
  IMeshMaterialVariableFactoryMng* m_variable_factory_mng = nullptr;
  std::unique_ptr<ObserverPool> m_observer_pool;
  String m_data_compressor_service_name;

  AllCell2AllEnvCell* m_allcell_2_allenvcell;

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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
