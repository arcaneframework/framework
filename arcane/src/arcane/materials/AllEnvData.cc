// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllEnvData.cc                                               (C) 2000-2023 */
/*                                                                           */
/* Informations sur les valeurs des milieux.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FunctorUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableBuildInfo.h"

#include "arcane/materials/ComponentItemListBuilder.h"
#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"

#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/AllEnvData.h"
#include "arcane/materials/internal/MaterialModifierOperation.h"
#include "arcane/materials/internal/ComponentConnectivityList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

class AllEnvData::IncrementalOneMaterialModifier
: public TraceAccessor
{
 public:

  IncrementalOneMaterialModifier(AllEnvData* all_env_data)
  : TraceAccessor(all_env_data->traceMng())
  , m_all_env_data(all_env_data)
  , m_material_mng(all_env_data->m_material_mng)
  {
  }

 public:

  void apply(MaterialModifierOperation* operation);

 private:

  AllEnvData* m_all_env_data = nullptr;
  MeshMaterialMng* m_material_mng = nullptr;

 private:

  void _switchComponentItemsForEnvironments(const IMeshEnvironment* modified_env, bool is_add_operation);
  void _switchComponentItemsForMaterials(const MeshMaterial* modified_mat, bool is_add);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvData::
AllEnvData(MeshMaterialMng* mmg)
: TraceAccessor(mmg->traceMng())
, m_material_mng(mmg)
, m_nb_env_per_cell(VariableBuildInfo(mmg->meshHandle(),mmg->name()+"_CellNbEnvironment"))
, m_item_internal_data(mmg)
{
  m_component_connectivity_list = new ComponentConnectivityList(m_material_mng);
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ALLENVDATA_DEBUG_LEVEL", true))
    m_verbose_debug_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvData::
~AllEnvData()
{
  delete m_component_connectivity_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
endCreate(bool is_continue)
{
  m_item_internal_data.endCreate();
  m_component_connectivity_list->endCreate(is_continue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AllEnvData::
_isFullVerbose() const
{
  return (m_verbose_debug_level >1 || traceMng()->verbosityLevel()>=5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_computeNbEnvAndNbMatPerCell()
{
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  // Calcule le nombre de milieux par maille, et pour chaque
  // milieu le nombre de matériaux par maille
  m_nb_env_per_cell.fill(0);
  for( MeshEnvironment* env : true_environments ){
    CellGroup cells = env->cells();
    ENUMERATE_CELL(icell,cells){
      ++m_nb_env_per_cell[icell];
    }
    env->computeNbMatPerCell();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_computeAndResizeEnvItemsInternal()
{
  // Calcule le nombre de milieux par maille, et pour chaque
  // milieu le nombre de matériaux par maille
  IMesh* mesh = m_material_mng->mesh();
  const IItemFamily* cell_family = mesh->cellFamily();
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  Integer nb_env = true_environments.size();
  Integer total_env_cell = 0;
  Integer total_mat_cell = 0;
  info(4) << "NB_ENV = " << nb_env;
  for (const MeshEnvironment* env : true_environments) {
    CellGroup cells = env->cells();
    Integer env_nb_cell = cells.size();
    info(4) << "NB_CELL=" << env_nb_cell << " env_name=" << cells.name();
    total_env_cell += env_nb_cell;
    total_mat_cell += env->totalNbCellMat();
  }

  // Il faut ajouter les infos pour les mailles de type AllEnvCell
  Int32 max_local_id = cell_family->maxLocalId();
  info(4) << "TOTAL_ENV_CELL=" << total_env_cell
          << " TOTAL_MAT_CELL=" << total_mat_cell;

  // TODO:
  // Le m_nb_mat_per_cell ne doit pas se faire sur les variablesIndexer().
  // Il doit prendre être different suivant les milieux et les matériaux.
  // - Si un milieu est le seul dans la maille, il prend la valeur globale.
  // - Si un matériau est le seul dans la maille, il prend la valeur
  // de la maille milieu correspondante (globale ou partielle suivant le cas)

  // Redimensionne les tableaux des infos
  // ATTENTION : ils ne doivent plus être redimensionnés par la suite sous peine
  // de tout invalider.
  m_item_internal_data.resizeNbAllEnvCell(max_local_id);
  m_item_internal_data.resizeNbEnvCell(total_env_cell);

  info(4) << "RESIZE all_env_items_internal size=" << max_local_id
          << " total_env_cell=" << total_env_cell;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_rebuildMaterialsAndEnvironmentsFromGroups()
{
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());
  const bool is_full_verbose = _isFullVerbose();

  for( const MeshEnvironment* env : true_environments ){
    MeshMaterialVariableIndexer* var_indexer = env->variableIndexer();
    ComponentItemListBuilder list_builder(var_indexer,0);
    CellGroup cells = var_indexer->cells();
    Integer var_nb_cell = cells.size();
    info(4) << "ENV_INDEXER (V2) i=" << var_indexer->index() << " NB_CELL=" << var_nb_cell << " name=" << cells.name()
            << " index=" << var_indexer->index();
    if (is_full_verbose)
      info(5) << "ENV_INDEXER (V2) name=" << cells.name() << " cells=" << cells.view().localIds();

    ENUMERATE_CELL(icell,cells){
      if (m_nb_env_per_cell[icell]>1)
        list_builder.addPartialItem(icell.itemLocalId());
      else
        // Je suis le seul milieu de la maille donc je prends l'indice global
        list_builder.addPureItem(icell.itemLocalId());
    }
    if (is_full_verbose)
      info() << "MAT_NB_MULTIPLE_CELL (V2) mat=" << var_indexer->name()
             << " nb_in_global=" << list_builder.pureMatVarIndexes().size()
             << " (ids=" << list_builder.pureMatVarIndexes() << ")"
             << " nb_in_multiple=" << list_builder.partialMatVarIndexes().size()
             << " (ids=" << list_builder.partialLocalIds() << ")";
    var_indexer->endUpdate(list_builder);
  }

  for( MeshEnvironment* env : true_environments )
    env->computeItemListForMaterials(m_nb_env_per_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_computeInfosForEnvCells()
{
  IMesh* mesh = m_material_mng->mesh();
  IItemFamily* cell_family = mesh->cellFamily();
  ItemGroup all_cells = cell_family->allItems();
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  ArrayView<ComponentItemInternal> all_env_items_internal = m_item_internal_data.allEnvItemsInternal();
  ArrayView<ComponentItemInternal> env_items_internal = m_item_internal_data.envItemsInternal();

  // Calcule pour chaque maille sa position dans le tableau des milieux
  // en considérant que les milieux de chaque maille sont rangés consécutivement
  // dans m_env_items_internal.
  Int32UniqueArray env_cell_indexes(cell_family->maxLocalId());
  {
    Integer env_cell_index = 0;
    ENUMERATE_CELL(icell,all_cells){
      Int32 lid = icell.itemLocalId();
      Int32 nb_env = m_nb_env_per_cell[icell];
      env_cell_indexes[lid] = env_cell_index;
      env_cell_index += nb_env;
    }
  }

  // Positionne les infos pour les EnvCell
  {
    Int32UniqueArray current_pos(env_cell_indexes);
    ItemInfoListView items_internal(cell_family);
    for( MeshEnvironment* env : true_environments ){
      const MeshMaterialVariableIndexer* var_indexer = env->variableIndexer();
      CellGroup cells = env->cells();
      Int32ConstArrayView nb_mat_per_cell = env->m_nb_mat_per_cell.asArray();

      env->resizeItemsInternal(var_indexer->nbItem());

      info(4) << "COMPUTE (V2) env_cells env=" << env->name() << " nb_cell=" << cells.size()
              << " index=" << var_indexer->index()
              << " max_multiple_index=" << var_indexer->maxIndexInMultipleArray();

      ConstArrayView<MatVarIndex> matvar_indexes = var_indexer->matvarIndexes();

      ArrayView<ComponentItemInternal*> env_items_internal_pointer = env->itemsInternalView();
      Int32ConstArrayView local_ids = var_indexer->localIds();

      for( Integer z=0, nb_id = matvar_indexes.size(); z<nb_id; ++z){
        MatVarIndex mvi = matvar_indexes[z];

        Int32 lid = local_ids[z];
        Int32 pos = current_pos[lid];
        ++current_pos[lid];
        Int32 nb_mat = nb_mat_per_cell[lid];
        ComponentItemInternal& ref_ii = env_items_internal[pos];
        env_items_internal_pointer[z] = &env_items_internal[pos];
        ref_ii.setSuperAndGlobalItem(&all_env_items_internal[lid],items_internal[lid]);
        ref_ii.setNbSubItem(nb_mat);
        ref_ii.setVariableIndex(mvi);
        ref_ii.setLevel(LEVEL_ENVIRONMENT);
      }
    }
    for( MeshEnvironment* env : true_environments ){
      env->computeMaterialIndexes(&m_item_internal_data);
    }
  }

  // Positionne les infos pour les AllEnvCell.
  {
    ENUMERATE_CELL(icell,all_cells){
      Cell c = *icell;
      Int32 lid = icell.itemLocalId();
      Int32 n = m_nb_env_per_cell[icell];
      ComponentItemInternal& ref_ii = all_env_items_internal[lid];
      ref_ii.setSuperAndGlobalItem(nullptr,c);
      ref_ii.setVariableIndex(MatVarIndex(0,lid));
      ref_ii.setNbSubItem(n);
      ref_ii.setLevel(LEVEL_ALLENVIRONMENT);
      if (n!=0)
        ref_ii.setFirstSubItem(&env_items_internal[env_cell_indexes[lid]]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie la cohérence des localIds() du variableIndexer().
 * avec la maille globale associée au milieu
 */
void AllEnvData::
_checkLocalIdsCoherency() const
{
  for (MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    Int32 index = 0;
    Int32ConstArrayView indexer_local_ids = env->variableIndexer()->localIds();
    ENUMERATE_COMPONENTCELL (icitem, env) {
      ComponentCell cc = *icitem;
      Int32 matvar_lid = cc.globalCell().localId();
      Int32 direct_lid = indexer_local_ids[index];
      if (matvar_lid != direct_lid)
        ARCANE_FATAL("Incoherent localId() matvar_lid={0} direct_lid={1} index={2}",
                     matvar_lid, direct_lid, index);
      ++index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remise à jour des structures suite à une modification des mailles
 * de matériaux ou de milieux.
 *
 * Cette méthode reconstruit les informations uniquement à partir des
 * groupes d'entités associés aux matériaux et milieux. Les variables
 * matériaux ne sont pas prise en compte par cette méthode et il est donc
 * possible qu'elles soient invalidées suite à cet appel. Si on souhaite la conservation
 * des valeurs, il faut d'abord sauvegarder les valeurs partielles,
 * appliquer cette méthode puis restaurer les valeurs partielles. La classe
 * MeshMaterialBackup permet cette sauvegarde/restauration.
 *
 * A noter que cette méthode peut être utilisée en reprise en conservant
 * les valeurs des variables matériaux car la structure des groupes
 * est la même après une reprise ce qui n'invalide pas les valeurs partielles.
 */
void AllEnvData::
forceRecompute(bool compute_all)
{
  m_material_mng->incrementTimestamp();

  ConstArrayView<MeshMaterialVariableIndexer*> vars_idx = m_material_mng->variablesIndexer();
  Integer nb_var = vars_idx.size();
  info(4) << "ForceRecompute NB_VAR_IDX=" << nb_var << " compute_all?=" << compute_all;

  const bool is_verbose_debug = m_verbose_debug_level >0;

  // Il faut compter le nombre total de mailles par milieu et par matériau

  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  if (compute_all)
    _computeNbEnvAndNbMatPerCell();

  // Calcul le nombre de milieux par maille, et pour chaque
  // milieu le nombre de matériaux par maille
  _computeAndResizeEnvItemsInternal();

  bool is_full_verbose = _isFullVerbose();

  if (compute_all)
    _rebuildMaterialsAndEnvironmentsFromGroups();

  for( const MeshEnvironment* env : true_environments ){
    const MeshMaterialVariableIndexer* var_indexer = env->variableIndexer();
    CellGroup cells = var_indexer->cells();
    Integer var_nb_cell = cells.size();
    info(4) << "FINAL_INDEXER i=" << var_indexer->index() << " NB_CELL=" << var_nb_cell << " name=" << cells.name()
            << " index=" << var_indexer->index();
    if (is_full_verbose){
      Int32UniqueArray my_array(cells.view().localIds());
      info(5) << "FINAL_INDEXER (V2) name=" << cells.name() << " cells=" << my_array;
      info(4) << "FINAL_MAT_NB_MULTIPLE_CELL (V2) mat=" << var_indexer->name()
             << " ids=" << var_indexer->matvarIndexes();
    }
  }

  // Initialise à des valeurs invalides pour détecter les erreurs.
  m_item_internal_data.resetEnvItemsInternal();

  _computeInfosForEnvCells();

  if (is_verbose_debug){
    _printAllEnvCells(m_material_mng->mesh()->allCells().view());
    for( IMeshMaterial* material : m_material_mng->materials() ){
      ENUMERATE_COMPONENTITEM(MatCell,imatcell,material){
        MatCell pmc = *imatcell;
        info() << "CELL IN MAT vindex=" << pmc._varIndex();
      }
    }
  }

  for( MeshEnvironment* env : true_environments ){
    env->componentData()->_rebuildPartData();
    for( MeshMaterial* mat : env->trueMaterials() )
      mat->componentData()->_rebuildPartData();
  }

  m_material_mng->checkValid();

  m_material_mng->syncVariablesReferences();

  if (is_verbose_debug){
    OStringStream ostr;
    m_material_mng->dumpInfos2(ostr());
    info() << ostr.str();
  }

  // Vérifie la cohérence des localIds() du variableIndexer()
  // avec la maille globale associée au milieu
  if (arcaneIsCheck())
    _checkLocalIdsCoherency();

  // Met à jour le AllCellToAllEnvCell s'il a été initialisé si la fonctionnalité est activé
  if (m_material_mng->isCellToAllEnvCellForRunCommand()) {
    auto* all_cell_to_all_env_cell(m_material_mng->getAllCellToAllEnvCell());
    if (all_cell_to_all_env_cell)
      all_cell_to_all_env_cell->bruteForceUpdate(m_material_mng->mesh()->allCells().internal()->itemsLocalId());
    else
      m_material_mng->createAllCellToAllEnvCell(platform::getDefaultDataAllocator());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
recomputeIncremental()
{
  forceRecompute(false);
  _checkConnectivityCoherency();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_printAllEnvCells(CellVectorView ids)
{
  ENUMERATE_ALLENVCELL(iallenvcell,m_material_mng->view(ids)){
    AllEnvCell all_env_cell = *iallenvcell;
    Integer cell_nb_env = all_env_cell.nbEnvironment();
    Cell cell = all_env_cell.globalCell();
    info() << "CELL2 uid=" << ItemPrinter(cell)
           << " nb_env=" << m_nb_env_per_cell[cell]
           << " direct_nb_env=" << cell_nb_env;
    for( Integer z=0; z<cell_nb_env; ++z ){
      EnvCell ec = all_env_cell.cell(z);
      Integer cell_nb_mat = ec.nbMaterial();
      info() << "CELL3 nb_mat=" << cell_nb_mat << " env_id=" << ec.environmentId();
      for( Integer k=0; k<cell_nb_mat; ++k ){
        MatCell mc = ec.cell(k);
        info() << "CELL4 mat_item=" << mc._varIndex() << " mat_id=" << mc.materialId();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Transforme les entités pour un milieu.
 *
 * Parcours le milieux \a env et
 * convertie les mailles pures en mailles partielles ou
 * inversement. Après conversion, les valeurs correspondantes aux
 * mailles modifiées sont mises à jour pour chaque variable.
 *
 * Si \a pure_to_partial est vrai, alors on transforme de pure en partiel
 * (dans le cas d'ajout de matériau) sinon on transforme de partiel en pure (dans le cas
 * de suppression d'un matériau)
 */
void AllEnvData::
_switchComponentItemsForMaterials(const MeshMaterial* modified_mat,bool is_add)
{
  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;

  Int32ArrayView cells_nb_env = m_nb_env_per_cell.asArray();

  bool is_verbose = traceMng()->verbosityLevel()>=5;

  for( MeshEnvironment* true_env : m_material_mng->trueEnvironments() ){
    for( MeshMaterial* mat : true_env->trueMaterials() ){
      // Ne traite pas le matériau en cours de modification.
      if (mat==modified_mat)
        continue;

      pure_local_ids.clear();
      partial_indexes.clear();

      const MeshEnvironment* env = mat->trueEnvironment();

      MeshMaterialVariableIndexer* indexer = mat->variableIndexer();
      ConstArrayView<Int32> cells_nb_mat = env->m_nb_mat_per_cell.asArray();

      info(4) << "TransformCells (V2) is_add?=" << is_add
              << " indexer=" << indexer->name();

      indexer->transformCells(cells_nb_env,cells_nb_mat,pure_local_ids,
                              partial_indexes,is_add,false,is_verbose);

      info(4) << "NB_MAT_TRANSFORM=" << pure_local_ids.size()
              << " name=" << mat->name();

      _copyBetweenPartialsAndGlobals(pure_local_ids,partial_indexes,
                                     indexer->index(),is_add);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Transforme les entités pour les milieux.
 *
 * Parcours les milieux, sauf le milieu modifié \a modified_env et
 * pour chacun convertie les mailles pures en mailles partielles ou
 * inversement. Après conversion, les valeurs correspondantes aux
 * mailles modifiées sont mises à jour pour chaque variable.
 *
 * Si \a pure_to_partial est vrai, alors on transforme de pure en partiel
 * (dans le cas d'ajout de matériau) sinon on transforme de partiel en pure (dans le cas
 * de suppression d'un matériau)
 */
void AllEnvData::
_switchComponentItemsForEnvironments(const IMeshEnvironment* modified_env,bool is_add_operation)
{
  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;

  Int32ArrayView cells_nb_env = m_nb_env_per_cell.asArray();

  bool is_verbose = traceMng()->verbosityLevel()>=5;

  for( const MeshEnvironment* env : m_material_mng->trueEnvironments() ){
    // Ne traite pas le milieu en cours de modification.
    if (env==modified_env)
      continue;

    pure_local_ids.clear();
    partial_indexes.clear();

    MeshMaterialVariableIndexer* indexer = env->variableIndexer();
    Int32ArrayView cells_nb_mat; // pas utilisé pour les milieux.

    info(4) << "TransformCells (V2) is_add?=" << is_add_operation
            << " indexer=" << indexer->name();

    indexer->transformCells(cells_nb_env,cells_nb_mat,pure_local_ids,
                            partial_indexes,is_add_operation,true,is_verbose);

    info(4) << "NB_ENV_TRANSFORM=" << pure_local_ids.size()
            << " name=" << env->name();

    _copyBetweenPartialsAndGlobals(pure_local_ids,partial_indexes,
                                   indexer->index(),is_add_operation);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie entre les valeurs partielles et les valeurs globales.
 *
 * Si \a pure_to_partial est vrai, alors on copie les valeurs globales
 * vers les valeurs partielles, sinon on fait l'inverse.
 * de suppression d'un matériau)
 */
void AllEnvData::
_copyBetweenPartialsAndGlobals(Int32ConstArrayView pure_local_ids,
                               Int32ConstArrayView partial_indexes,
                               Int32 indexer_index,bool is_add_operation)
{
  if (pure_local_ids.empty())
    return;

  // Comme on a modifié des mailles, il faut mettre à jour les valeurs
  // correspondantes pour chaque variable.
  //info(4) << "NB_TRANSFORM=" << nb_transform << " name=" << e->name();
  //Integer indexer_index = indexer->index();
  auto func = [=](IMeshMaterialVariable* mv){
    if (is_add_operation)
      mv->_copyGlobalToPartial(indexer_index,pure_local_ids,partial_indexes);
    else
      mv->_copyPartialToGlobal(indexer_index,pure_local_ids,partial_indexes);
  };
  functor::apply(m_material_mng,&MeshMaterialMng::visitVariables,func);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
updateMaterialDirect(MaterialModifierOperation* operation)
{
  // Vérifie dans le cas des mailles à ajouter si elles ne sont pas déjà
  // dans le matériau et dans le cas des mailles à supprimer si elles y sont.
  if (arcaneIsCheck())
    operation->filterIds();

  _updateMaterialDirect(operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
updateMaterialIncremental(MaterialModifierOperation* operation)
{
  // Vérifie dans le cas des mailles à ajouter si elles ne sont pas déjà
  // dans le matériau et dans le cas des mailles à supprimer si elles y sont.
  if (arcaneIsCheck())
    operation->filterIds();

  IncrementalOneMaterialModifier modifier(this);
  modifier.apply(operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_updateMaterialDirect(MaterialModifierOperation* operation)
{
  bool is_add = operation->isAdd();
  IMeshMaterial* mat = operation->material();
  Int32ConstArrayView ids = operation->ids();

  auto* true_mat = ARCANE_CHECK_POINTER(dynamic_cast<MeshMaterial*>(mat));

  info(4) << "Using optimisation updateMaterialDirect operation=" << operation;

  const IMeshEnvironment* env = mat->environment();
  MeshEnvironment* true_env = true_mat->trueEnvironment();
  Integer nb_mat = env->nbMaterial();

  Int32UniqueArray cells_changed_in_env;

  if (nb_mat!=1){

    // S'il est possible d'avoir plusieurs matériaux par milieu, il faut gérer
    // pour chaque maille si le milieu évolue suite à l'ajout/suppression de matériau.
    // Les deux cas sont :
    // - en cas d'ajout, le milieu évolue pour une maille s'il n'y avait pas
    //   de matériau avant. Dans ce cas le milieu est ajouté à la maille.
    // - en cas de suppression, le milieu évolue dans la maille s'il y avait
    //   1 seul matériau avant. Dans ce cas le milieu est supprimé de la maille.

    Int32UniqueArray cells_unchanged_in_env;
    Int32ArrayView cells_nb_mat = true_env->m_nb_mat_per_cell.asArray();
    const Int32 ref_nb_mat = is_add ? 0 : 1;

    info(4) << "Using optimisation updateMaterialDirect is_add?=" << is_add;

    for( Integer i=0, n=ids.size(); i<n; ++i ){
      Int32 lid = ids[i];
      if (cells_nb_mat[lid]!=ref_nb_mat){
        info(5)<< "CELL i=" << i << " lid="<< lid << " unchanged in environment nb_mat=" << cells_nb_mat[lid];
        cells_unchanged_in_env.add(lid);
      }
      else{
        cells_changed_in_env.add(lid);
      }
    }

    Integer nb_unchanged_in_env = cells_unchanged_in_env.size();
    info(4) << "Cells unchanged in environment n=" << nb_unchanged_in_env;

    if (is_add){
      mat->cells().addItems(cells_unchanged_in_env);
    }
    else{
      mat->cells().removeItems(cells_unchanged_in_env);
    }
    true_env->updateItemsDirect(m_nb_env_per_cell,true_mat,cells_unchanged_in_env,is_add,false);

    ids = cells_changed_in_env.view();
  }

  Int32ArrayView cells_nb_env = m_nb_env_per_cell.asArray();

  // Met à jour le nombre de milieux de chaque maille.
  if (is_add){
    for( Int32 id : ids )
      ++cells_nb_env[id];
  }
  else{
    for( Int32 id : ids )
      --cells_nb_env[id];
  }

  // Comme on a ajouté/supprimé des mailles matériau dans le milieu,
  // il faut transformer les mailles pures en mailles partielles (en cas
  // d'ajout) ou les mailles partielles en mailles pures (en cas de
  // suppression).
  info(4) << "Transform PartialPure for material name=" << true_mat->name();
  _switchComponentItemsForMaterials(true_mat,is_add);
  info(4) << "Transform PartialPure for environment name=" << env->name();
  _switchComponentItemsForEnvironments(env,is_add);

  // Si je suis mono-mat, alors mat->cells()<=>env->cells() et il ne faut
  // mettre à jour que l'un des deux groupes.
  bool need_update_env = (nb_mat!=1);

  if (is_add){
    mat->cells().addItems(ids);
    if (need_update_env)
      env->cells().addItems(ids);
  }
  else{
    mat->cells().removeItems(ids);
    if (need_update_env)
      env->cells().removeItems(ids);
  }
  true_env->updateItemsDirect(m_nb_env_per_cell,true_mat,ids,is_add,need_update_env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_checkConnectivityCoherency()
{
  info() << "AllEnvData: checkCoherency()";
  const VariableCellInt16& nb_env_v2 = m_component_connectivity_list->cellNbEnvironment();

  ItemGroup all_cells = m_material_mng->mesh()->allCells();

  Int32 nb_error = 0;
  ENUMERATE_CELL(icell,all_cells){
    Int32 ref_value = m_nb_env_per_cell[icell];
    Int32 current_value = nb_env_v2[icell];
    if (ref_value!=current_value){
      ++nb_error;
      if (nb_error<10)
        error() << "Invalid for nb_environment cell=" << icell->uniqueId()
                << " ref_value=" << ref_value << " current=" << current_value;
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Invalid values for number of environments nb_error={0}",nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::IncrementalOneMaterialModifier::
apply(MaterialModifierOperation* operation)
{
  bool is_add = operation->isAdd();
  IMeshMaterial* mat = operation->material();
  Int32ConstArrayView ids = operation->ids();

  auto* true_mat = ARCANE_CHECK_POINTER(dynamic_cast<MeshMaterial*>(mat));

  info(4) << "Using optimisation updateMaterialDirect operation=" << operation;

  const IMeshEnvironment* env = mat->environment();
  MeshEnvironment* true_env = true_mat->trueEnvironment();
  Integer nb_mat = env->nbMaterial();

  ComponentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();

  Int32UniqueArray cells_changed_in_env;

  if (nb_mat!=1){

    // S'il est possible d'avoir plusieurs matériaux par milieu, il faut gérer
    // pour chaque maille si le milieu évolue suite à l'ajout/suppression de matériau.
    // Les deux cas sont :
    // - en cas d'ajout, le milieu évolue pour une maille s'il n'y avait pas
    //   de matériau avant. Dans ce cas le milieu est ajouté à la maille.
    // - en cas de suppression, le milieu évolue dans la maille s'il y avait
    //   1 seul matériau avant. Dans ce cas le milieu est supprimé de la maille.

    Int32UniqueArray cells_unchanged_in_env;
    Int32ArrayView cells_nb_mat = true_env->m_nb_mat_per_cell.asArray();
    const Int32 ref_nb_mat = is_add ? 0 : 1;

    info(4) << "Using optimisation updateMaterialDirect is_add?=" << is_add;

    for( Integer i=0, n=ids.size(); i<n; ++i ){
      Int32 lid = ids[i];
      if (cells_nb_mat[lid]!=ref_nb_mat){
        info(5)<< "CELL i=" << i << " lid="<< lid << " unchanged in environment nb_mat=" << cells_nb_mat[lid];
        cells_unchanged_in_env.add(lid);
      }
      else{
        cells_changed_in_env.add(lid);
      }
    }

    Integer nb_unchanged_in_env = cells_unchanged_in_env.size();
    info(4) << "Cells unchanged in environment n=" << nb_unchanged_in_env;

    if (is_add){
      mat->cells().addItems(cells_unchanged_in_env);
    }
    else{
      mat->cells().removeItems(cells_unchanged_in_env);
    }
    true_env->updateItemsDirect(m_all_env_data->m_nb_env_per_cell,true_mat,cells_unchanged_in_env,is_add,false);

    // Met à jour le nombre de matériaux de chaque maille.
    {
      Int16 mat_id = true_mat->componentId();
      if (is_add){
        connectivity->addCellsToMaterial(mat_id,ids);
      }
      else{
        connectivity->removeCellsToMaterial(mat_id,ids);
      }
    }
    ids = cells_changed_in_env.view();
  }

  Int32ArrayView cells_nb_env = m_all_env_data->m_nb_env_per_cell.asArray();

  // Met à jour le nombre de milieux de chaque maille.
  {
    Int16 env_id = true_env->componentId();
    if (is_add){
      connectivity->addCellsToEnvironment(env_id,ids);
      for( Int32 id : ids )
        ++cells_nb_env[id];
    }
    else{
      connectivity->removeCellsToEnvironment(env_id,ids);
      for( Int32 id : ids )
        --cells_nb_env[id];
    }
  }

  // Comme on a ajouté/supprimé des mailles matériau dans le milieu,
  // il faut transformer les mailles pures en mailles partielles (en cas
  // d'ajout) ou les mailles partielles en mailles pures (en cas de
  // suppression).
  info(4) << "Transform PartialPure for material name=" << true_mat->name();
  _switchComponentItemsForMaterials(true_mat,is_add);
  info(4) << "Transform PartialPure for environment name=" << env->name();
  _switchComponentItemsForEnvironments(env,is_add);

  // Si je suis mono-mat, alors mat->cells()<=>env->cells() et il ne faut
  // mettre à jour que l'un des deux groupes.
  bool need_update_env = (nb_mat!=1);

  if (is_add){
    mat->cells().addItems(ids);
    if (need_update_env)
      env->cells().addItems(ids);
  }
  else{
    mat->cells().removeItems(ids);
    if (need_update_env)
      env->cells().removeItems(ids);
  }
  true_env->updateItemsDirect(m_all_env_data->m_nb_env_per_cell,true_mat,ids,is_add,need_update_env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Transforme les entités pour un milieu.
 *
 * Parcours le milieux \a env et
 * convertie les mailles pures en mailles partielles ou
 * inversement. Après conversion, les valeurs correspondantes aux
 * mailles modifiées sont mises à jour pour chaque variable.
 *
 * Si \a is_add est vrai, alors on transforme de pure en partiel
 * (ajout de matériau) sinon on transforme de partiel en pure
 * (suppression d'un matériau)
 */
void AllEnvData::IncrementalOneMaterialModifier::
_switchComponentItemsForMaterials(const MeshMaterial* modified_mat,bool is_add)
{
  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;

  Int32ArrayView cells_nb_env = m_all_env_data->m_nb_env_per_cell.asArray();

  bool is_verbose = traceMng()->verbosityLevel()>=5;

  for( MeshEnvironment* true_env : m_material_mng->trueEnvironments() ){
    for( MeshMaterial* mat : true_env->trueMaterials() ){
      // Ne traite pas le matériau en cours de modification.
      if (mat==modified_mat)
        continue;

      pure_local_ids.clear();
      partial_indexes.clear();

      const MeshEnvironment* env = mat->trueEnvironment();

      MeshMaterialVariableIndexer* indexer = mat->variableIndexer();
      ConstArrayView<Int32> cells_nb_mat = env->m_nb_mat_per_cell.asArray();

      info(4) << "TransformCells (V2) is_add?=" << is_add
              << " indexer=" << indexer->name();

      indexer->transformCells(cells_nb_env,cells_nb_mat,pure_local_ids,
                              partial_indexes,is_add,false,is_verbose);

      info(4) << "NB_MAT_TRANSFORM=" << pure_local_ids.size()
              << " name=" << mat->name();

      m_all_env_data->_copyBetweenPartialsAndGlobals(pure_local_ids,partial_indexes,
                                                     indexer->index(),is_add);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Transforme les entités pour les milieux.
 *
 * Parcours les milieux, sauf le milieu modifié \a modified_env et
 * pour chacun convertie les mailles pures en mailles partielles ou
 * inversement. Après conversion, les valeurs correspondantes aux
 * mailles modifiées sont mises à jour pour chaque variable.
 *
 * Si \a pure_to_partial est vrai, alors on transforme de pure en partiel
 * (dans le cas d'ajout de matériau) sinon on transforme de partiel en pure (dans le cas
 * de suppression d'un matériau)
 */
void AllEnvData::IncrementalOneMaterialModifier::
_switchComponentItemsForEnvironments(const IMeshEnvironment* modified_env,bool is_add_operation)
{
  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;

  Int32ArrayView cells_nb_env = m_all_env_data->m_nb_env_per_cell.asArray();

  bool is_verbose = traceMng()->verbosityLevel()>=5;

  for( const MeshEnvironment* env : m_material_mng->trueEnvironments() ){
    // Ne traite pas le milieu en cours de modification.
    if (env==modified_env)
      continue;

    pure_local_ids.clear();
    partial_indexes.clear();

    MeshMaterialVariableIndexer* indexer = env->variableIndexer();
    Int32ArrayView cells_nb_mat; // pas utilisé pour les milieux.

    info(4) << "TransformCells (V2) is_add?=" << is_add_operation
            << " indexer=" << indexer->name();

    indexer->transformCells(cells_nb_env,cells_nb_mat,pure_local_ids,
                            partial_indexes,is_add_operation,true,is_verbose);

    info(4) << "NB_ENV_TRANSFORM=" << pure_local_ids.size()
            << " name=" << env->name();

    m_all_env_data->_copyBetweenPartialsAndGlobals(pure_local_ids,partial_indexes,
                                                   indexer->index(),is_add_operation);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
