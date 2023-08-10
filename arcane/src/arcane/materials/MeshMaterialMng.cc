// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialMng.cc                                          (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire des matériaux et milieux d'un maillage.                      */
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

#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MeshMaterialSynchronizer.h"
#include "arcane/materials/MeshMaterialVariableSynchronizer.h"
#include "arcane/materials/MeshMaterialExchangeMng.h"
#include "arcane/materials/EnumeratorTracer.h"
#include "arcane/materials/MeshMaterialVariableFactoryRegisterer.h"
#include "arcane/materials/internal/AllEnvData.h"
#include "arcane/materials/internal/MeshMaterialModifierImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file MaterialsGlobal.h
 *
 * Liste des déclarations globales pour les matériaux.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO:
 * - Vérifier qu'on ne créé qu'une seule instance de MeshModifier.
 * - Vérifier par exemple dans synchronizeMaterialsInCells()
 * qu'on n'est pas en train de modifier le maillage.
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
arcaneCreateMeshMaterialMng(const MeshHandle& mesh_handle,const String& name)
{
  MeshMaterialMng* mmm = new MeshMaterialMng(mesh_handle,name);
  //std::cout << "CREATE MESH_MATERIAL_MNG mesh_name=" << mesh_handle.meshName()
  //          << " ref=" << mesh_handle.reference() << " this=" << mmm << "\n";
  mmm->build();
  return mmm;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialMng::
MeshMaterialMng(const MeshHandle& mesh_handle,const String& name)
// TODO: utiliser le ITraceMng du maillage. Le faire lors de l'init
: TraceAccessor(mesh_handle.traceMng())
, m_mesh_handle(mesh_handle)
, m_internal_api(new InternalApi(this))
, m_variable_mng(mesh_handle.variableMng())
, m_name(name)
{
  m_modifier = new MeshMaterialModifierImpl(this);
  m_all_env_data = new AllEnvData(this);
  m_exchange_mng = new MeshMaterialExchangeMng(this);
  m_variable_factory_mng = arcaneCreateMeshMaterialVariableFactoryMng(this);
  m_observer_pool = std::make_unique<ObserverPool>();
  m_observer_pool->addObserver(this,&MeshMaterialMng::_onMeshDestroyed,mesh_handle.onDestroyObservable());

  String s = platform::getEnvironmentVariable("ARCANE_ALLENVCELL_FOR_RUNCOMMAND");
  if (!s.null())
    m_is_allcell_2_allenvcell = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialMng::
~MeshMaterialMng()
{
  //std::cout << "DESTROY MESH MATERIAL MNG this=" << this << '\n';
  IEnumeratorTracer* tracer = IEnumeratorTracer::singleton();
  if (tracer)
    tracer->dumpStats();

  delete m_variable_factory_mng;
  delete m_exchange_mng;
  delete m_all_cells_env_only_synchronizer;
  delete m_all_cells_mat_env_synchronizer;
  delete m_all_env_data;
  delete m_properties;

  for( MeshMaterial* m : m_true_materials )
    delete m;
  m_true_materials.clear();

  for( MeshEnvironment* e : m_true_environments )
    delete e;
  m_true_environments.clear();

  for( IMeshBlock* b : m_true_blocks )
    delete b;

  for( MeshMaterialInfo* mmi : m_materials_info )
    delete mmi;

  for( MeshMaterialVariableIndexer* mvi : m_variables_indexer_to_destroy )
    delete mvi;

  m_modifier->dumpStats();
  delete m_modifier;

  delete m_internal_api;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
build()
{
  {
    auto* x = MeshMaterialVariableFactoryRegisterer::firstRegisterer();
    while (x){
      m_variable_factory_mng->registerFactory(x->createFactory());
      x = x->nextRegisterer();
    }
  }

  // Choix des optimisations.
  {
    int default_flags = 0;

    //GG: Décommenter pour activer par défaut les optimisations.
    //default_flags = (int)eModificationFlags::GenericOptimize | (int)eModificationFlags::OptimizeMultiAddRemove;

    int opt_flag_value = 0;
    String env_name = "ARCANE_MATERIAL_MODIFICATION_FLAGS";
    String opt_flag_str = platform::getEnvironmentVariable(env_name);
    if (!opt_flag_str.null()){
      if (builtInGetValue(opt_flag_value,opt_flag_str)){
        pwarning() << "Invalid value '" << opt_flag_str
                   << " 'for environment variable '" << env_name
                   << "'";
        opt_flag_value = default_flags;
      }
    }
    else{
      opt_flag_value = default_flags;
    }
    m_modification_flags = opt_flag_value;
  }

  // Choix de la version de l'implémentation des synchronisations
  {
    String env_name = "ARCANE_MATSYNCHRONIZE_VERSION";
    String env_value = platform::getEnvironmentVariable(env_name);
    info() << "ENV_VALUE=" << env_value;
    Integer version = m_synchronize_variable_version;
    if (!env_value.null()){
      if (builtInGetValue(version,env_value)){
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

  // Choix du service de compression
  {
    String env_name = "ARCANE_MATERIAL_DATA_COMPRESSOR_NAME";
    String env_value = platform::getEnvironmentVariable(env_name);
    if (!env_value.null()){
      info() << "Use serivice '" << env_value << "' for material data compression";
      m_data_compressor_service_name = env_value;
    }
  }


  m_exchange_mng->build();
  // Si les traces des énumérateurs sur les entités sont actives, active celles
  // sur les matériaux.
  // TODO: rendre ce code thread-safe en cas d'utilisation de IParallelMng via les threads
  // et ne l'appeler qu'une fois.
  IItemEnumeratorTracer* item_tracer = IItemEnumeratorTracer::singleton();
  if (item_tracer){
    info() << "Adding material enumerator tracing";
    EnumeratorTracer::_setSingleton(new EnumeratorTracer(traceMng(),item_tracer->perfCounterRef()));
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
 * \brief Création d'un matériau.
 *
 * Créé un matériau de nom \a name, dans le milieu \a env, avec les
 * infos \a infos.
 */
MeshMaterial* MeshMaterialMng::
_createMaterial(MeshEnvironment* env,MeshMaterialInfo* infos,const String& name)
{
  _checkEndCreate();
  if (infos->materialMng()!=this)
    ARCANE_FATAL("Invalid materialMng() for material info");
  if (env->materialMng()!=this)
    ARCANE_FATAL("Invalid materialMng() for environment");
  Integer var_index = m_variables_indexer.size();
  Int16 mat_id = CheckedConvert::toInt16(m_materials.size());
  MeshMaterial* mat = new MeshMaterial(infos,env,name,mat_id);
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
  // Vérifie que le matériau n'est pas déjà enregistré.
  MeshMaterialInfo* old_mmi = _findMaterialInfo(name);
  if (old_mmi)
    ARCANE_FATAL("Un matériau de nom '{0}' est déjà enregistré",name);

  MeshMaterialInfo* mmi = new MeshMaterialInfo(this,name);
  m_materials_info.add(mmi);
  return mmi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Création d'un milieu.
 *
 * Les infos du milieu sont données par la structure \a infos.
 * En même temps que le milieu sont créés tous les matériaux
 * le constituant.
 */
IMeshEnvironment* MeshMaterialMng::
createEnvironment(const MeshEnvironmentBuildInfo& infos)
{
  _checkEndCreate();
  Int16 env_index = CheckedConvert::toInt16(m_environments.size());
  // Vérifie qu'un milieu de même nom n'existe pas.
  const String& env_name = infos.name();
  MeshEnvironment* old_me = _findEnvironment(env_name);
  if (old_me)
    ARCANE_FATAL("Un milieu de nom '{0}' est déjà enregistré",env_name);

  info() << "Creating environment name=" << env_name << " index=" << env_index;
  // Créé le milieu
  MeshEnvironment* me = new MeshEnvironment(this,env_name,env_index);
  me->build();
  m_true_environments.add(me);
  m_environments.add(me);
  m_environments_as_components.add(me);

  // Créé et ajoute les matériaux
  Integer nb_mat = infos.materials().size();
  for( Integer i=0; i<nb_mat; ++i ){
    const MeshEnvironmentBuildInfo::MatInfo& buildinfo = infos.materials()[i];
    const String& mat_name = buildinfo.m_name;
    String new_mat_name = env_name + "_" + mat_name;
    MeshMaterialInfo* mat_info = _findMaterialInfo(mat_name);
    if (!mat_info){
      ARCANE_FATAL("Aucun matériau de nom '{0}' n'est défini",mat_name);
    }
    MeshMaterial* mm = _createMaterial(me,mat_info,new_mat_name);
    me->addMaterial(mm);
    mat_info->_addEnvironment(env_name);
  }
  // Si le milieu contient plusieurs matériaux, il faut lui allouer
  // des valeurs partielles. Sinon, ses valeurs partielles sont celles
  // de son unique matériau.
  {
    MeshMaterialVariableIndexer* var_idx = nullptr;
    if (nb_mat==1){
      var_idx = me->materials()[0]->_internalApi()->variableIndexer();
    }
    else{
      var_idx = new MeshMaterialVariableIndexer(traceMng(),me->name());
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
  // Vérifie qu'un bloc de même nom n'existe pas.
  const String& name = infos.name();
  const MeshBlock* old_mb = _findBlock(name);
  if (old_mb)
    ARCANE_FATAL("Un bloc de nom '{0}' est déjà enregistré",name);

  info() << "Creating block name=" << name << " index=" << block_index
         << " nb_env=" << infos.environments().size();
  Integer nb_env = infos.environments().size();
  for( Integer i=0; i<nb_env; ++i )
    info() << " Adding environment name=" << infos.environments()[i]->name() << " to block";

  // Créé le bloc
  MeshBlock* mb = new MeshBlock(this,block_index,infos);
  mb->build();
  m_true_blocks.add(mb);
  m_blocks.add(mb);

  return mb;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
addEnvironmentToBlock(IMeshBlock* block,IMeshEnvironment* env)
{
  MeshBlock* mb = ARCANE_CHECK_POINTER(dynamic_cast<MeshBlock*>(block));
  mb->addEnvironment(env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
removeEnvironmentToBlock(IMeshBlock* block,IMeshEnvironment* env)
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

  m_modifier->initOptimizationFlags();

  m_all_env_data->endCreate(is_continue);

  auto synchronizer = mesh()->cellFamily()->allItemsSynchronizer();
  m_all_cells_mat_env_synchronizer = new MeshMaterialVariableSynchronizer(this,synchronizer,MatVarSpace::MaterialAndEnvironment);
  m_all_cells_env_only_synchronizer = new MeshMaterialVariableSynchronizer(this,synchronizer,MatVarSpace::Environment);

  // Détermine la liste de tous les composants.
  {
    Integer nb_component = m_environments_as_components.size() + m_materials_as_components.size();
    m_components.reserve(nb_component);
    m_components.addRange(m_environments_as_components);
    m_components.addRange(m_materials_as_components);
  }

  // Il faut construire et initialiser les variables qui ont été
  // créées avant cette allocation.
  for( const auto& i : m_full_name_variable_map ){
    IMeshMaterialVariable* mv = i.second;
    info(4) << "BUILD FROM MANAGER name=" << mv->name() << " this=" << this;
    mv->buildFromManager(is_continue);
  }
  if (is_continue)
    _endUpdate();
  m_is_end_create = true;

  // Vérifie que les milieux sont valides.
  // NOTE: on ne peut pas toujours appeler checkValid()
  // (notamment au démarrage) car les groupes d'entités existent,
  // mais les infos matériaux associées ne sont pas
  // forcément encore créés.
  // (Il faudra regarder une si cela est dû au mode compatible ou pas).
  for( IMeshEnvironment* env : m_environments ){
    env->checkValid();
  }

  // Maintenant que tout est créé, il est valide d'enregistrer les mécanismes
  // d'échange.
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

void  MeshMaterialMng::
setDataCompressorServiceName(const String& name)
{
  m_data_compressor_service_name = name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialModifierImpl* MeshMaterialMng::
modifier()
{
  return m_modifier;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialInfo* MeshMaterialMng::
_findMaterialInfo(const String& name)
{
  for( MeshMaterialInfo* mmi : m_materials_info )
    if (mmi->name()==name)
      return mmi;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshEnvironment* MeshMaterialMng::
findEnvironment(const String& name,bool throw_exception)
{
  IMeshEnvironment* env = _findEnvironment(name);
  if (env)
    return env;
  if (throw_exception)
    ARCANE_FATAL("No environment named '{0}'",name);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshEnvironment* MeshMaterialMng::
_findEnvironment(const String& name)
{
  for( MeshEnvironment* env : m_true_environments )
    if (env->name()==name)
      return env;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshBlock* MeshMaterialMng::
findBlock(const String& name,bool throw_exception)
{
  IMeshBlock* block = _findBlock(name);
  if (block)
    return block;
  if (throw_exception)
    ARCANE_FATAL("No block named '{0}'",name);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshBlock* MeshMaterialMng::
_findBlock(const String& name)
{
  for( MeshBlock* b : m_true_blocks )
    if (b->name()==name)
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
 * \brief Remise à jour des structures suite à une modification des mailles
 * de matériaux ou de milieux.
 */
void MeshMaterialMng::
_endUpdate()
{
  m_all_env_data->forceRecompute(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Met à jour les références des variables.
 *
 * Cela doit être fait lorsque le nombre d'éléments par matériau ou milieu
 * change car les tableaux contenant les variables associées peuvent être
 * modifiés lors de l'opération.
 */
void MeshMaterialMng::
syncVariablesReferences()
{
  for( const auto& i : m_full_name_variable_map ){
    IMeshMaterialVariable* mv = i.second;
    info(4) << "SYNC REFERENCES FROM MANAGER name=" << mv->name();
    mv->syncReferences();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
visitVariables(IFunctorWithArgumentT<IMeshMaterialVariable*>* functor)
{
  if (!functor)
    return;
  for( const auto& i : m_full_name_variable_map ){
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
  const VariableCellInt32 nb_env_per_cell = m_all_env_data->nbEnvPerCell();
  ENUMERATE_ALLENVCELL(iallenvcell,view(all_cells.view().localIds())){
    AllEnvCell all_env_cell = *iallenvcell;
    Integer cell_nb_env = all_env_cell.nbEnvironment();
    Cell cell = all_env_cell.globalCell();
    Int64 cell_uid = cell.uniqueId();
    if (all_env_cell.level()!=LEVEL_ALLENVIRONMENT)
      ARCANE_FATAL("Bad level for all_env_item");

    if (all_env_cell.globalCell()!=cell)
      ARCANE_FATAL("Bad corresponding globalCell() in all_env_item");
    if (cell_nb_env!=nb_env_per_cell[cell])
      ARCANE_FATAL("Bad value for nb_env direct='{0}' var='{1}'",
                   cell_nb_env,nb_env_per_cell[cell]);
    
    for( Integer z=0; z<cell_nb_env; ++z ){
      EnvCell ec = all_env_cell.cell(z);
      Integer cell_nb_mat = ec.nbMaterial();
      ComponentItemInternal* eii = ec.internal();
      if (all_env_cell.internal()!=eii->superItem())
        ARCANE_FATAL("Bad corresponding allEnvItem() in env_item uid={0}",cell_uid);
      if (eii->globalItemBase()!=cell)
        ARCANE_FATAL("Bad corresponding globalItem() in env_item");
      if (eii->level()!=LEVEL_ENVIRONMENT)
        ARCANE_FATAL("Bad level '{0}' for in env_item",eii->level());
      // Si la maille n'est pas pure, la variable milieu ne peut être équivalente à
      // la variable globale.
      if (cell_nb_env>1 && ec._varIndex().arrayIndex()==0)
        ARCANE_FATAL("Global index for a partial cell env_item={0}",ec);

      for( Integer k=0; k<cell_nb_mat; ++k ){
        MatCell mc = ec.cell(k);
        ComponentItemInternal* mci = mc.internal();
        if (eii!=mci->superItem())
          ARCANE_FATAL("Bad corresponding env_item in mat_item");
        if (mci->globalItemBase()!=cell)
          ARCANE_FATAL("Bad corresponding globalItem() in mat_item");
        if (mci->level()!=LEVEL_MATERIAL)
          ARCANE_FATAL("Bad level '{0}' for in mat_item",mci->level());
        // Si la maille n'est pas pure, la variable matériau ne peut être équivalente à
        // la variable globale.
        if ((cell_nb_env>1 || cell_nb_mat>1) && mc._varIndex().arrayIndex()==0){
          ARCANE_FATAL("Global index for a partial cell matitem={0} name={1} nb_mat={2} nb_env={3}",
                       mc,mc.material()->name(),cell_nb_mat,cell_nb_env);
        }
      }
    }
  }

  for( IMeshEnvironment* env : m_environments ){
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

  // Recherche la variable globale de nom \a name
  // et si on la trouve, prend son nom complet pour
  // la variable matériau.
  const IVariable* global_var = m_variable_mng->findMeshVariable(mesh(),name);
  if (global_var){
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
  if (i!=m_full_name_variable_map.end())
    return i->second;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialVariable* MeshMaterialMng::
checkVariable(IVariable* global_var)
{
  auto i = m_var_to_mat_var_map.find(global_var);
  if (i!=m_var_to_mat_var_map.end())
    return i->second;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
fillWithUsedVariables(Array<IMeshMaterialVariable*>& variables)
{
  variables.clear();

  // Utilise la map sur les noms des variables pour garantir un même
  // ordre de parcours quels que soient les sous-domaines.
  for( const auto& i : m_full_name_variable_map ){
    IMeshMaterialVariable* ivar = i.second;
    if (ivar->globalVariable()->isUsed())
      variables.add(ivar);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
addVariable(IMeshMaterialVariable* var)
{
  //TODO: le verrou m_variable_lock doit etre actif.
  IVariable* gvar = var->globalVariable();
  info(4) << "MAT_ADD_VAR global_var=" << gvar << " var=" << var << " this=" << this;
  m_var_to_mat_var_map.insert(std::make_pair(gvar,var));
  m_full_name_variable_map.insert(std::make_pair(gvar->fullName(),var));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
removeVariable(IMeshMaterialVariable* var)
{
  //TODO: le verrou m_variable_lock doit etre actif.
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
  for( IMeshMaterial* mat : m_materials ){
    o << "--   Materiau name=" << mat->name() << '\n';
  }

  o << "-- Liste des milieux\n";
  for( IMeshEnvironment* me : m_environments ){
    ConstArrayView<IMeshMaterial*> env_materials = me->materials();
    const MeshMaterialVariableIndexer* env_var_idx = me->_internalApi()->variableIndexer();
    Integer nb_env_mat = env_materials.size();
    o << "--   Milieu name=" << me->name()
      << " nb_mat=" << nb_env_mat
      << " nb_cell=" << me->cells().size()
      << " var_idx = " << env_var_idx->index()
      << " ids=" << env_var_idx->matvarIndexes()
      << '\n';
    for( IMeshMaterial* mm : env_materials ){
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
// TODO: fusionner dumpInfos2() et dumpInfo().
void MeshMaterialMng::
dumpInfos2(std::ostream& o)
{
  Integer nb_mat = m_materials.size();
  Integer nb_env = m_environments.size();
  Integer nb_var_idx = m_variables_indexer.size();
  o << "-- Material and Environment infos: nb_env=" << nb_env
    << " nb_mat=" << nb_mat << " timestamp=" << m_timestamp
    << " nb_var_idx=" << nb_var_idx
    << "\n";
  Integer nb_cell = mesh()->allCells().size();
  if (nb_cell!=0){
    Integer nb_pure_env = 0;
    const VariableCellInt32& nb_env_per_cell = m_all_env_data->nbEnvPerCell();
    ENUMERATE_CELL(icell,mesh()->allCells()){
      if (nb_env_per_cell[icell]<=1)
        ++nb_pure_env;
    }
    o << " nb_cell=" << nb_cell << " nb_pure_env=" << nb_pure_env
      << " nb_partial=" << (nb_cell-nb_pure_env)
      << " percent=" << (100*nb_pure_env)/nb_cell
      << "\n";
  }

  o << "-- Liste des milieux\n";
  for( MeshEnvironment* me : m_true_environments ){
    ConstArrayView<IMeshMaterial*> env_materials = me->materials();
    const MeshMaterialVariableIndexer* env_var_idx = me->variableIndexer();
    Integer nb_env_mat = env_materials.size();
    Integer nb_env_cell = me->cells().size();
    Integer nb_pure_mat = 0;
    if (nb_env_mat>1){
      const VariableCellInt32& nb_mat_per_cell = me->m_nb_mat_per_cell;
      ENUMERATE_CELL(icell,me->cells()){
        if (nb_mat_per_cell[icell]<=1)
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
    if (nb_env_cell!=0)
      o << " percent=" << (nb_pure_mat*100)/nb_env_cell;
    o << '\n';
    for( Integer j=0; j<nb_env_mat; ++j ){
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
  MeshMaterialSynchronizer mms(this);
  return mms.synchronizeMaterialsInCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
checkMaterialsInCells(Integer max_print)
{
  MeshMaterialSynchronizer mms(this);
  mms.checkMaterialsInCells(max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
dumpCellInfos(Cell cell,std::ostream& o)
{
  CellToAllEnvCellConverter all_env_cell_converter(this);
  AllEnvCell all_env_cell = all_env_cell_converter[cell];
  Cell global_cell = all_env_cell.globalCell();
  o << "Cell uid=" << ItemPrinter(global_cell) << '\n';
  ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
    o << "ENV name=" << (*ienvcell).environment()->name()
      << " component_idx=" << ComponentItemLocalId(ienvcell) << '\n';
    ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
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
  return CellToAllEnvCellConverter(m_all_env_data->allEnvItemsInternal());
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
view(Int32ConstArrayView local_ids)
{
  return AllEnvCellVectorView(local_ids,m_all_env_data->allEnvItemsInternal());
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
  Ref<IMeshMaterialMng> getTrueReference(const MeshHandle& mesh_handle,bool is_create) override;
 public:
  static MeshMaterialMngFactory m_mesh_material_mng_factory;
};

MeshMaterialMngFactory MeshMaterialMngFactory::m_mesh_material_mng_factory{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMeshMaterialMng> MeshMaterialMngFactory::
getTrueReference(const MeshHandle& mesh_handle,bool is_create)
{
  //TODO: faire lock pour multi-thread
  typedef AutoDestroyUserData<Ref<IMeshMaterialMng>> UserDataType;
  
  const char* name = "MeshMaterialMng_StdMat";
  IUserDataList* udlist = mesh_handle.meshUserDataList();

  IUserData* ud = udlist->data(name,true);
  if (!ud){
    if (!is_create)
      return {};
    IMeshMaterialMng* mm = arcaneCreateMeshMaterialMng(mesh_handle,"StdMat");
    Ref<IMeshMaterialMng> mm_ref = makeRef(mm);
    udlist->setData(name,new UserDataType(new Ref<IMeshMaterialMng>(mm_ref)));
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
  m_properties = new Properties(*(mesh()->properties()),String("MeshMaterialMng_")+name());
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

  // Sauve le numéro de version pour être certain que c'est OK en reprise
  m_properties->set("Version",SERIALIZE_VERSION);

  // Sauve dans les propriétés les infos nécessaires pour recréer les
  // matériaux et milieux.
  UniqueArray<String> material_info_names;
  for( MeshMaterialInfo* mat_info : m_materials_info ){
    material_info_names.add(mat_info->name());
  }
  m_properties->set("MaterialInfoNames",material_info_names);

  UniqueArray<String> env_names;
  UniqueArray<Int32> env_nb_mat;
  UniqueArray<String> env_mat_names;
  ENUMERATE_ENV(ienv,this){
    IMeshEnvironment* env = *ienv;
    env_names.add(env->name());
    info(5) << "SAVE ENV_NAME name=" << env->name() << " nb_mat=" << env->nbMaterial();
    env_nb_mat.add(env->nbMaterial());
    ENUMERATE_MAT(imat,env){
      const String& name = (*imat)->infos()->name();
      info(5) << "SAVE MAT_NAME name=" << name;
      env_mat_names.add(name);
    }
  }
  m_properties->set("EnvNames",env_names);
  m_properties->set("EnvNbMat",env_nb_mat);
  m_properties->set("EnvMatNames",env_mat_names);

  // Sauve les infos nécessaires pour les block.
  // Pour chaque bloc, son nom et le nom du groupe de maille correspondant.
  UniqueArray<String> block_names;
  UniqueArray<String> block_cell_group_names;
  UniqueArray<Int32> block_nb_env;
  UniqueArray<String> block_env_names;
  for( IMeshBlock* block : m_blocks ){
    block_names.add(block->name());
    block_cell_group_names.add(block->cells().name());
    block_nb_env.add(block->nbEnvironment());
    ENUMERATE_ENV(ienv,block){
      const String& name = (*ienv)->name();
      info(5) << "SAVE BLOCK ENV_NAME name=" << name;
      block_env_names.add(name);
    }
  }
  m_properties->set("BlockNames",block_names);
  m_properties->set("BlockCellGroupNames",block_cell_group_names);
  m_properties->set("BlockNbEnv",block_nb_env);
  m_properties->set("BlockEnvNames",block_env_names);
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

  // Sauve le numéro de version pour être sur que c'est OK en reprise
  Int32 v = m_properties->getInt32("Version");
  if (v!=SERIALIZE_VERSION)
    ARCANE_FATAL("Bad serializer version: trying to read from incompatible checkpoint v={0} expected={1}",
                 v,SERIALIZE_VERSION);

  UniqueArray<String> material_info_names;
  m_properties->get("MaterialInfoNames",material_info_names);
  for( const String& mat_name : material_info_names )
    this->registerMaterialInfo(mat_name);

  UniqueArray<String> env_names;
  UniqueArray<Int32> env_nb_mat;
  UniqueArray<String> env_mat_names;
  m_properties->get("EnvNames",env_names);
  m_properties->get("EnvNbMat",env_nb_mat);
  m_properties->get("EnvMatNames",env_mat_names);

  Integer mat_index = 0;
  for( Integer ienv=0, nenv=env_names.size(); ienv<nenv; ++ienv ){
    Materials::MeshEnvironmentBuildInfo env_build(env_names[ienv]);
    Integer nb_mat = env_nb_mat[ienv];
    for( Integer imat=0; imat<nb_mat; ++imat ){
      env_build.addMaterial(env_mat_names[mat_index]);
      ++mat_index;
    }
    this->createEnvironment(env_build);
  }

  // Recréé les blocs.
  // Pour chaque bloc, son nom et le nom du groupe de maille correspondant.
  UniqueArray<String> block_names;
  UniqueArray<String> block_cell_group_names;
  UniqueArray<String> block_env_names;
  UniqueArray<Int32> block_nb_env;
  m_properties->get("BlockNames",block_names);
  m_properties->get("BlockCellGroupNames",block_cell_group_names);
  m_properties->get("BlockNbEnv",block_nb_env);
  m_properties->get("BlockEnvNames",block_env_names);
  const IItemFamily* cell_family = mesh()->cellFamily();
  Integer block_env_index = 0;
  for( Integer i=0, n=block_names.size(); i<n; ++i ){
    String name = block_names[i];
    String cell_group_name = block_cell_group_names[i];
    CellGroup cells = cell_family->findGroup(cell_group_name);
    if (cells.null())
      ARCANE_FATAL("Can not find cell group '{0}' for block creation",
                   cell_group_name);
    MeshBlockBuildInfo mbbi(name,cells);
    if (!block_nb_env.empty()){
      Integer nb_env = block_nb_env[i];
      for( Integer ienv=0; ienv<nb_env; ++ienv ){
        const String& name2 = block_env_names[block_env_index];
        ++block_env_index;
        IMeshEnvironment* env = findEnvironment(name2,false);
        if (!env)
          ARCANE_FATAL("Invalid environment name '{0}' for recreating blocks",name2);
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
  // Il faut détruire cette instance ici car elle a besoin de IItemFamily
  // dans son destructeur et il est possible qu'il n'y ait plus de famille
  // si le destructeur de IMeshMaterialMng est appelé après la destruction
  // du maillage (ce qui peut arriver en C# par exemple).
  delete m_exchange_mng;
  m_exchange_mng = nullptr;

  _unregisterAllVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialMng::
_unregisterAllVariables()
{
  // Recopie dans un tableau toutes les références.
  // Il faut le faire avant les appels à unregisterVariable()
  // car ces derniers modifient la liste chainée des références
  UniqueArray<MeshMaterialVariableRef*> m_all_refs;

  for( const auto& i : m_full_name_variable_map ){
    const IMeshMaterialVariable* var = i.second;

    for( MeshMaterialVariableRef::Enumerator iref(var); iref.hasNext(); ++iref ){
      MeshMaterialVariableRef* ref = *iref;
      m_all_refs.add(ref);
    }
  }

  for( MeshMaterialVariableRef* ref : m_all_refs )
    ref->unregisterVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
