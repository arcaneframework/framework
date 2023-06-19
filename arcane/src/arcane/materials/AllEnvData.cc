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

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"

#include "arcane/VariableBuildInfo.h"

#include "arcane/materials/AllEnvData.h"
#include "arcane/materials/MeshMaterialMng.h"
#include "arcane/materials/ComponentItemListBuilder.h"
#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvData::
AllEnvData(MeshMaterialMng* mmg)
: TraceAccessor(mmg->traceMng())
, m_material_mng(mmg)
, m_nb_env_per_cell(VariableBuildInfo(mmg->meshHandle(),mmg->name()+"_CellNbEnvironment"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvData::
~AllEnvData()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_computeNbEnvAndNbMatPerCell()
{
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  // Calcul le nombre de milieux par maille, et pour chaque
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

  IMesh* mesh = m_material_mng->mesh();
  IItemFamily* cell_family = mesh->cellFamily();
  ItemGroup all_cells = cell_family->allItems();
  ConstArrayView<MeshMaterialVariableIndexer*> vars_idx = m_material_mng->variablesIndexer();
  Integer nb_var = vars_idx.size();
  info(4) << "ForceRecompute NB_VAR_IDX=" << nb_var
          << " compute_all?=" << compute_all;

  //TODO: Utiliser celui de MeshMaterialMng
  const bool is_verbose = false;

  // Il faut compter le nombre total de mailles par milieu et par matériau

  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  if (compute_all)
    _computeNbEnvAndNbMatPerCell();

  // Calcul le nombre de milieux par maille, et pour chaque
  // milieu le nombre de matériaux par maille
  Integer nb_env = true_environments.size();
  Integer total_env_cell = 0;
  Integer total_mat_cell = 0;
  info(4) << "NB_ENV = " << nb_env;
  for( MeshEnvironment* env : true_environments ){
    CellGroup cells = env->cells();
    Integer env_nb_cell = cells.size();
    info(4) << "NB_CELL=" << env_nb_cell << " env_name=" << cells.name();
    total_env_cell += env_nb_cell;
    total_mat_cell += env->totalNbCellMat();
  }

  // Il faut ajouter les infos pour les mailles de type AllEnvCell
  Integer max_local_id = cell_family->maxLocalId();
  info(4) << "TOTAL_ENV_CELL=" << total_env_cell
          << " TOTAL_MAT_CELL=" << total_mat_cell;
  //TODO:
  // Le m_nb_mat_per_cell ne doit pas se faire sur les variablesIndexer().
  // Il doit prendre etre different suivant les milieux et les materiaux.
  // - Si un milieu est le seul dans la maille, il prend la valeur globale.
  // - Si un materiau est le seul dans la maille, il prend la valeur
  // de la maille milieu correspondante (globale ou partielle suivant le cas)

  // Redimensionne les tableaux des infos
  // ATTENTION: il ne doivent plus être redimensionnés par la suite sous peine
  // de tout invalider.
  m_all_env_items_internal.resize(max_local_id);
  m_env_items_internal.resize(total_env_cell);
  info(4) << "RESIZE all_env_items_internal size=" << max_local_id
         << " total_env_cell=" << total_env_cell;

  bool is_full_verbose = traceMng()->verbosityLevel()>=5;

  if (compute_all){
    for( Integer i=0; i<nb_env; ++i ){
      MeshMaterialVariableIndexer* var_indexer = true_environments[i]->variableIndexer();
      ComponentItemListBuilder list_builder(var_indexer,0);
      CellGroup cells = var_indexer->cells();
      Integer var_nb_cell = cells.size();
      info(4) << "ENV_INDEXER (V2) i=" << i << " NB_CELL=" << var_nb_cell << " name=" << cells.name()
              << " index=" << var_indexer->index();
      if (is_full_verbose){
        Int32UniqueArray my_array(cells.view().localIds());
        info(5) << "ENV_INDEXER (V2) name=" << cells.name() << " cells=" << my_array;
      }
      ENUMERATE_CELL(icell,var_indexer->cells()){
        if (m_nb_env_per_cell[icell]>1)
          list_builder.addPartialItem(icell.itemLocalId());
        else
          // Je suis le seul milieu de la maille donc je prend l'indice global
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

    for( Integer i=0; i<nb_env; ++i ){
      true_environments[i]->computeItemListForMaterials(m_nb_env_per_cell);
    }
  }

  for( Integer i=0; i<nb_env; ++i ){
    MeshMaterialVariableIndexer* var_indexer = true_environments[i]->variableIndexer();
    CellGroup cells = var_indexer->cells();
    Integer var_nb_cell = cells.size();
    info(4) << "FINAL_INDEXER i=" << i << " NB_CELL=" << var_nb_cell << " name=" << cells.name()
            << " index=" << var_indexer->index();
    if (is_full_verbose){
      Int32UniqueArray my_array(cells.view().localIds());
      info(5) << "FINAL_INDEXER (V2) name=" << cells.name() << " cells=" << my_array;
      info(4) << "FINAL_MAT_NB_MULTIPLE_CELL (V2) mat=" << var_indexer->name()
             << " ids=" << var_indexer->matvarIndexes();
    }
  }

  // Initialise à des valeurs invalides pour détecter les
  // erreurs.
  {
    for( Integer i=0; i<max_local_id; ++i ){
      ComponentItemInternal& ref_ii = m_all_env_items_internal[i];
      ref_ii.reset();
    }
    for( Integer i=0; i<total_env_cell; ++i ){
      ComponentItemInternal& ref_ii = m_env_items_internal[i];
      ref_ii.reset();
    }
  }


  // Calcule pour chaque maille sa position dans le tableau des milieux
  // en considérant que les milieux de chaque maille sont rangés consécutivement
  // dans m_env_items_internal.
  Int32UniqueArray env_cell_indexes(max_local_id);
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
    for( Integer i=0; i<nb_env; ++i ){
      MeshEnvironment* env = true_environments[i];
      MeshMaterialVariableIndexer* var_indexer = env->variableIndexer();
      CellGroup cells = env->cells();
      Int32ConstArrayView nb_mat_per_cell = env->m_nb_mat_per_cell.asArray();

      env->resizeItemsInternal(var_indexer->nbItem());

      info(4) << "COMPUTE (V2) env_cells env=" << env->name() << " nb_cell=" << cells.size()
              << " index=" << var_indexer->index()
              << " max_multiple_index=" << var_indexer->maxIndexInMultipleArray();

      ConstArrayView<MatVarIndex> matvar_indexes = var_indexer->matvarIndexes();

      ArrayView<ComponentItemInternal*> env_items_internal = env->itemsInternalView();
      Int32ConstArrayView local_ids = var_indexer->localIds();

      for( Integer z=0, nb_id = matvar_indexes.size(); z<nb_id; ++z){
        MatVarIndex mvi = matvar_indexes[z];

        Int32 lid = local_ids[z];
        Int32 pos = current_pos[lid];
        ++current_pos[lid];
        Int32 nb_mat = nb_mat_per_cell[lid];
        ComponentItemInternal& ref_ii = m_env_items_internal[pos];
        env_items_internal[z] = &m_env_items_internal[pos];
        ref_ii.setSuperAndGlobalItem(&m_all_env_items_internal[lid],items_internal[lid]);
        ref_ii.setNbSubItem(nb_mat);
        ref_ii.setVariableIndex(mvi);
        ref_ii.setLevel(LEVEL_ENVIRONMENT);
      }
    }
    for( Integer i=0; i<nb_env; ++i ){
      MeshEnvironment* env = true_environments[i];
      env->computeMaterialIndexes();
    }
  }

  // Positionne les infos pour les AllEnvCell.
  {
    ENUMERATE_CELL(icell,all_cells){
      Cell c = *icell;
      Int32 lid = icell.itemLocalId();
      Int32 nb_env = m_nb_env_per_cell[icell];
      ComponentItemInternal& ref_ii = m_all_env_items_internal[lid];
      ref_ii.setSuperAndGlobalItem(0,c);
      ref_ii.setVariableIndex(MatVarIndex(0,lid));
      ref_ii.setNbSubItem(nb_env);
      ref_ii.setLevel(LEVEL_ALLENVIRONMENT);
      if (nb_env!=0)
        ref_ii.setFirstSubItem(&m_env_items_internal[env_cell_indexes[lid]]);
    }
  }


  if (is_verbose){
    printAllEnvCells(all_cells.view().localIds());
  }

  if (is_verbose){
    for( IMeshMaterial* material : m_material_mng->materials() ){
      //Integer nb_mat = m_materials.size();
      //for( Integer i=0; i<nb_mat; ++i ){
      //IMeshMaterial* material = m_materials[i];
      info() << "MAT name=" << material->name();
      ENUMERATE_COMPONENTITEM(MatCell,imatcell,material){
        MatCell pmc = *imatcell;
        info() << "CELL IN MAT vindex=" << pmc._varIndex();
      }
    }
  }

  for( MeshEnvironment* env : true_environments ){
    env->componentData()->rebuildPartData();
    for( MeshMaterial* mat : env->trueMaterials() )
      mat->componentData()->rebuildPartData();
  }

  m_material_mng->checkValid();

  m_material_mng->syncVariablesReferences();

  if (is_verbose){
    OStringStream ostr;
    m_material_mng->dumpInfos2(ostr());    info() << ostr.str();
  }

  // Vérifie la cohérence des localIds() du variableIndexer()
  // avec la maille globale associée au milieu
  const bool check_localid_coherency = false;
  if (check_localid_coherency){
    for( MeshEnvironment* env : true_environments ){
      Int32 index = 0;
      Int32ConstArrayView indexer_local_ids = env->variableIndexer()->localIds();
      ENUMERATE_COMPONENTCELL(icitem,env){
        ComponentCell cc = *icitem;
        Int32 matvar_lid = cc.globalCell().localId();
        Int32 direct_lid = indexer_local_ids[index];
        if (matvar_lid!=direct_lid)
          ARCANE_FATAL("Incoherent localId() matvar_lid={0} direct_lid={1} index={2}",
                       matvar_lid,direct_lid,index);
        ++index;
      }
    }
  }

  // Met à jour le AllCellToAllEnvCell s'il a été initialisé si la fonctionnalité est activé
  if (m_material_mng->isCellToAllEnvCellForRunCommand()) {
    auto* allCell2AllEnvCell(m_material_mng->getAllCellToAllEnvCell());
    if (allCell2AllEnvCell)
      allCell2AllEnvCell->bruteForceUpdate(m_material_mng->mesh()->allCells().internal()->itemsLocalId());
    else
      m_material_mng->createAllCellToAllEnvCell(platform::getDefaultDataAllocator());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
printAllEnvCells(Int32ConstArrayView ids)
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
_switchComponentItemsForMaterials(MeshMaterial* modified_mat,eOperation operation)
{
  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;

  Int32ArrayView cells_nb_env = m_nb_env_per_cell.asArray();

  bool is_verbose = traceMng()->verbosityLevel()>=5;

  for( MeshEnvironment* env : m_material_mng->trueEnvironments() ){
    for( MeshMaterial* mat : env->trueMaterials() ){
      // Ne traite pas le matériau en cours de modification.
      if (mat==modified_mat)
        continue;

      pure_local_ids.clear();
      partial_indexes.clear();

      MeshEnvironment* env = mat->trueEnvironment();

      MeshMaterialVariableIndexer* indexer = mat->variableIndexer();
      Int32ArrayView cells_nb_mat = env->m_nb_mat_per_cell.asArray();

      info(4) << "TransformCells (V2) operation=" << operation
              << " indexer=" << indexer->name();

      indexer->transformCells(cells_nb_env,cells_nb_mat,pure_local_ids,
                              partial_indexes,operation,false,is_verbose);

      info(4) << "NB_MAT_TRANSFORM=" << pure_local_ids.size()
              << " name=" << mat->name();

      _copyBetweenPartialsAndGlobals(pure_local_ids,partial_indexes,
                                     indexer->index(),operation);
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
_switchComponentItemsForEnvironments(IMeshEnvironment* modified_env,eOperation operation)
{
  UniqueArray<Int32> pure_local_ids;
  UniqueArray<Int32> partial_indexes;

  Int32ArrayView cells_nb_env = m_nb_env_per_cell.asArray();

  bool is_verbose = traceMng()->verbosityLevel()>=5;

  for( MeshEnvironment* env : m_material_mng->trueEnvironments() ){
    // Ne traite pas le milieu en cours de modification.
    if (env==modified_env)
      continue;

    pure_local_ids.clear();
    partial_indexes.clear();

    MeshMaterialVariableIndexer* indexer = env->variableIndexer();
    Int32ArrayView cells_nb_mat; // pas utilisé pour les milieux.

    info(4) << "TransformCells (V2) operation=" << operation
            << " indexer=" << indexer->name();

    indexer->transformCells(cells_nb_env,cells_nb_mat,pure_local_ids,
                            partial_indexes,operation,true,is_verbose);

    info(4) << "NB_ENV_TRANSFORM=" << pure_local_ids.size()
            << " name=" << env->name();

    _copyBetweenPartialsAndGlobals(pure_local_ids,partial_indexes,
                                   indexer->index(),operation);
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
                               Int32 indexer_index,eOperation operation)
{
  Integer nb_transform = pure_local_ids.size();
  if (nb_transform==0)
    return;

  // Comme on a modifié des mailles, il faut mettre à jour les valeurs
  // correspondantes pour chaque variable.
  //info(4) << "NB_TRANSFORM=" << nb_transform << " name=" << e->name();
  //Integer indexer_index = indexer->index();
  auto func = [=](IMeshMaterialVariable* mv){
    if (operation==eOperation::Add)
      mv->_copyGlobalToPartial(indexer_index,pure_local_ids,partial_indexes);
    else
      mv->_copyPartialToGlobal(indexer_index,pure_local_ids,partial_indexes);
  };
  functor::apply(m_material_mng,&MeshMaterialMng::visitVariables,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie si les mailles \a ids sont déjà dans le matériau \a mat.
 *
 * Si \a operation==eOperation::Add, vérifie que les mailles de \a ids
 * ne sont pas déjà dans le matériau et si \a operation==eOperation::Remove, vérifie
 * que les mailles de \a ids sont dans le matériau.
 *
 * Vérifie aussi qu'un élément n'est présent qu'une fois dans la liste \a ids.
 *
 * Retourne le nombre d'erreurs.
 */
Integer AllEnvData::
_checkMaterialPrescence(IMeshMaterial* mat,Int32ConstArrayView ids,eOperation operation)
{
  // TODO: faut-il vérifier la validité des \a ids
  //(ils sont compris entre 0 et max_loca_id-1) ?

  MeshMaterialVariableIndexer* indexer = mat->variableIndexer();
  IItemFamily* item_family = mat->cells().itemFamily();
  ItemInfoListView items_internal(item_family);
  Integer max_local_id = item_family->maxLocalId();
  UniqueArray<bool> presence_flags(max_local_id,false);
  Int32ConstArrayView mat_local_ids = indexer->localIds();
  Integer nb_error = 0;
  String name = mat->name();

  for( Integer i=0, n=ids.size(); i<n; ++i ){
    Int32 lid = ids[i];
    if (presence_flags[lid]){
      info() << "ERROR: item " << ItemPrinter(items_internal[lid])
             << " is present several times in add/remove list for material mat=" << name;
      ++nb_error;
    }
    presence_flags[lid] = true;
  }

  if (operation==eOperation::Add){
    for( Integer i=0, n=mat_local_ids.size(); i<n; ++i ){
      Int32 lid = mat_local_ids[i];
      if (presence_flags[lid]){
        info() << "ERROR: item " << ItemPrinter(items_internal[lid])
               << " is already in material mat=" << name;
        ++nb_error;
      }
    }
  }
  else if (operation==eOperation::Remove){
    for( Integer i=0, n=mat_local_ids.size(); i<n; ++i ){
      Int32 lid = mat_local_ids[i];
      presence_flags[lid]= false;
    }

    for( Integer i=0, n=ids.size(); i<n; ++i ){
      Int32 lid = ids[i];
      if (presence_flags[lid]){
        info() << "ERROR: item " << ItemPrinter(items_internal[lid])
               << " is not in material mat=" << name;
        ++nb_error;
      }
    }
  }
  else
    _throwBadOperation(operation);

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Filtre le tableau des mailles \a ids pour qu'il soit valide.
 *
 * Cette méthode permet de filtrer les valeurs de \a ids afin
 * qu'il ne reste que les valeurs valides pour qu'on puisse les ajouter
 * (si \a do_add est vrai) ou supprimer (si \a do_add est faux) du matériau
 * \a mat.
 *
 * Les valeurs valides sont stockées dans \a valid_ids.
 */
void AllEnvData::
_filterValidIds(IMeshMaterial* mat,Int32ConstArrayView ids,bool do_add,Int32Array& valid_ids)
{
  // TODO: faut-il vérifier la validité des \a ids
  //(ils sont compris entre 0 et max_loca_id-1) ?

  MeshMaterialVariableIndexer* indexer = mat->variableIndexer();
  IItemFamily* item_family = mat->cells().itemFamily();
  Integer max_local_id = item_family->maxLocalId();
  UniqueArray<bool> presence_flags(max_local_id,false);
  Int32ConstArrayView mat_local_ids = indexer->localIds();
  
  UniqueArray<Int32> unique_occurence_lids;
  unique_occurence_lids.reserve(ids.size());

  for( Integer i=0, n=ids.size(); i<n; ++i ){
    Int32 lid = ids[i];
    if (!presence_flags[lid]){
      unique_occurence_lids.add(lid);
      presence_flags[lid] = true;
    }
  }

  valid_ids.clear();

  if (do_add){
    for( Integer i=0, n=mat_local_ids.size(); i<n; ++i ){
      Int32 lid = mat_local_ids[i];
      if (presence_flags[lid]){
        ;
      }
      else
        valid_ids.add(lid);
    }
  }
  else{
    for( Integer i=0, n=mat_local_ids.size(); i<n; ++i ){
      Int32 lid = mat_local_ids[i];
      presence_flags[lid] = false;
    }

    for( Integer i=0, n=unique_occurence_lids.size(); i<n; ++i ){
      Int32 lid = unique_occurence_lids[i];
      if (presence_flags[lid]){
        ;
      }
      else
        valid_ids.add(lid);
    }
  }
  info(4) << "FILTERED_IDS n=" << valid_ids.size()
          << " ids=" << valid_ids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_throwBadOperation(eOperation operation)
{
  ARCANE_THROW(ArgumentException,"Invalid value for operation v={0}",(int)operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
updateMaterialDirect(IMeshMaterial* mat,Int32ConstArrayView ids,
                      eOperation operation)
{
  UniqueArray<Int32> filtered_ids;
  bool filter_invalid = true;
  // Vérifie dans le cas des mailles à ajouter si elles ne sont pas déjà
  // dans le matériau et dans le cas des mailles à supprimer si elles y sont.
  if (arcaneIsCheck()){
    Integer nb_error = _checkMaterialPrescence(mat,ids,operation);
    if (nb_error!=0){
      if (filter_invalid){
        _filterValidIds(mat,ids,true,filtered_ids);
        ids = filtered_ids.constView();
      }
      else
        ARCANE_FATAL("Invalid values for adding items in material name={0} nb_error={1}",
                     mat->name(),nb_error);
    }
  }

  _updateMaterialDirect(mat,ids,operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_updateMaterialDirect(IMeshMaterial* mat,Int32ConstArrayView ids,eOperation operation)
{
  if (operation!=eOperation::Add && operation!=eOperation::Remove)
    _throwBadOperation(operation);
  bool is_add = (operation==eOperation::Add);

  MeshMaterial* true_mat = dynamic_cast<MeshMaterial*>(mat);
  if (!true_mat)
    throw NotImplementedException(A_FUNCINFO,"material is not an instance of MeshMaterial");

  info(4) << "Using optimisation updateMaterialDirect operation=" << operation;

  IMeshEnvironment* env = mat->environment();
  MeshEnvironment* true_env = true_mat->trueEnvironment();
  Integer nb_mat = env->nbMaterial();

  Int32UniqueArray cells_changed_in_env;

  if (nb_mat!=1){

    // S'il est possible d'avoir plusieurs matériaux par milieu, il faut gérer
    // pour chaque maille si le milieu évolue suite à l'ajout/suppression de matériau.
    // les deux cas sont:
    // - en cas d'ajout, le milieu évolue pour une maille s'il n'y avait pas
    //   de matériau avant. Dans ce cas le milieu est ajouté à la maille.
    // - en cas de suppression, le milieu évolue dans la maille s'il y avait
    //   1 seul matériau avant. Dans ce cas le milieu est supprimé de la maille.

    Int32UniqueArray cells_unchanged_in_env;
    Int32ArrayView cells_nb_mat = true_env->m_nb_mat_per_cell.asArray();
    const Int32 ref_nb_mat = (is_add) ? 0 : 1;

    info(4) << "Using optimisation updateMaterialDirect with multimat operation="
            << operation;

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
    true_env->updateItemsDirect(m_nb_env_per_cell,true_mat,cells_unchanged_in_env,operation,false);

    ids = cells_changed_in_env.view();
  }

  Int32ArrayView cells_nb_env = m_nb_env_per_cell.asArray();

  // Met à jour le nombre de milieux de chaque maille.
  if (is_add){
    for( Integer i=0, n=ids.size(); i<n; ++i ){
      ++cells_nb_env[ids[i]];
    }
  }
  else{
    for( Integer i=0, n=ids.size(); i<n; ++i ){
      --cells_nb_env[ids[i]];
    }
  }

  // Comme on a ajouté/supprimé des mailles matériau dans le milieu,
  // il faut transformer les mailles pures en mailles partielles (en cas
  // d'ajout) ou les mailles partielles en mailles pures (en cas de
  // suppression).
  info(4) << "Transform PartialPure for material name=" << true_mat->name();
  _switchComponentItemsForMaterials(true_mat,operation);
  info(4) << "Transform PartialPure for environemnt name=" << env->name();
  _switchComponentItemsForEnvironments(env,operation);

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
  true_env->updateItemsDirect(m_nb_env_per_cell,true_mat,ids,operation,need_update_env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
