// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableIndexer.cc                              (C) 2000-2023 */
/*                                                                           */
/* Indexer pour les variables materiaux.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"
#include "arcane/materials/internal/ComponentModifierWorkInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableIndexer::
MeshMaterialVariableIndexer(ITraceMng* tm,const String& name)
: TraceAccessor(tm)
, m_index(-1)
, m_max_index_in_multiple_array(-1)
, m_name(name)
, m_matvar_indexes(platform::getAcceleratorHostMemoryAllocator())
, m_local_ids(platform::getAcceleratorHostMemoryAllocator())
, m_is_environment(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableIndexer::
MeshMaterialVariableIndexer(const MeshMaterialVariableIndexer& rhs)
: TraceAccessor(rhs)
, m_index(rhs.m_index)
, m_max_index_in_multiple_array(rhs.m_max_index_in_multiple_array)
, m_name(rhs.m_name)
, m_cells(rhs.m_cells)
, m_matvar_indexes(rhs.m_matvar_indexes)
, m_local_ids(rhs.m_local_ids)
, m_is_environment(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdate(const ComponentItemListBuilder& builder)
{
  ConstArrayView<MatVarIndex> pure_matvar = builder.pureMatVarIndexes();
  ConstArrayView<MatVarIndex> partial_matvar = builder.partialMatVarIndexes();

  Integer nb_pure = pure_matvar.size();
  Integer nb_partial = partial_matvar.size();
  m_matvar_indexes.resize(nb_pure+nb_partial);

  m_matvar_indexes.subView(0,nb_pure).copy(pure_matvar);
  m_matvar_indexes.subView(nb_pure,nb_partial).copy(partial_matvar);

  Int32ConstArrayView local_ids_in_multiple = builder.partialLocalIds();

  {
    m_local_ids.resize(nb_pure+nb_partial);
    Integer index = 0;
    for( Integer i=0, n=nb_pure; i<n; ++i ){
      m_local_ids[index] = pure_matvar[i].valueIndex();
      ++index;
    }
    for( Integer i=0, n=nb_partial; i<n; ++i ){
      m_local_ids[index] = local_ids_in_multiple[i];
      ++index;
    }
  }

  // NOTE: a priori, ici on est sur que m_max_index_in_multiple array vaut
  // nb_partial+1
  {
    Int32 max_index_in_multiple = (-1);
    for( Integer i=0; i<nb_partial; ++i ){
      max_index_in_multiple = math::max(partial_matvar[i].valueIndex(),max_index_in_multiple);
    }
    m_max_index_in_multiple_array = max_index_in_multiple;
  }

  info(4) << "END_UPDATE max_index=" << m_max_index_in_multiple_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdateAdd(const ComponentItemListBuilder& builder)
{
  ConstArrayView<MatVarIndex> pure_matvar = builder.pureMatVarIndexes();
  ConstArrayView<MatVarIndex> partial_matvar = builder.partialMatVarIndexes();

  Integer nb_pure_to_add = pure_matvar.size();
  Integer nb_partial_to_add = partial_matvar.size();
  Integer total_to_add = nb_pure_to_add + nb_partial_to_add;
  Integer current_nb_item = nbItem();

  m_matvar_indexes.resize(current_nb_item + total_to_add);

  m_matvar_indexes.subView(current_nb_item,nb_pure_to_add).copy(pure_matvar);
  m_matvar_indexes.subView(current_nb_item+nb_pure_to_add,nb_partial_to_add).copy(partial_matvar);

  Int32ConstArrayView local_ids_in_multiple = builder.partialLocalIds();

  {
    m_local_ids.resize(current_nb_item + total_to_add);
    Int32ArrayView local_ids_view = m_local_ids.subView(current_nb_item,total_to_add);
    Integer index = 0;
    for( Integer i=0, n=nb_pure_to_add; i<n; ++i ){
      local_ids_view[index] = pure_matvar[i].valueIndex();
      ++index;
    }
    for( Integer i=0, n=nb_partial_to_add; i<n; ++i ){
      local_ids_view[index] = local_ids_in_multiple[i];
      ++index;
    }
  }

  {
    Int32 max_index_in_multiple = m_max_index_in_multiple_array;
    for( Integer i=0; i<nb_partial_to_add; ++i ){
      max_index_in_multiple = math::max(partial_matvar[i].valueIndex(),max_index_in_multiple);
    }
    m_max_index_in_multiple_array = max_index_in_multiple;
  }
  info(4) << "END_UPDATE_ADD max_index=" << m_max_index_in_multiple_array
          << " nb_partial_to_add=" << nb_partial_to_add;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdateRemove(ConstArrayView<bool> removed_local_ids_filter,Integer nb_remove)
{
  // TODO: à supprimer et à remplacer par la version qui prend un
  // ComponentModifierWorkInfo à la place

  Integer nb_item = nbItem();
  Integer orig_nb_item = nb_item;

  //ATTENTION: on modifie nb_item pendant l'itération.
  for (Integer i=0; i<nb_item; ++i) {
    Int32 lid = m_local_ids[i];
    if (removed_local_ids_filter[lid]) {
      // Déplace le dernier MatVarIndex vers l'élément courant.
      Int32 last = nb_item - 1;
      m_matvar_indexes[i] = m_matvar_indexes[last];
      m_local_ids[i] = m_local_ids[last];
      //info() << "REMOVE ITEM lid=" << lid << " i=" << i;
      --nb_item;
      --i; // Il faut refaire l'itération courante.
    }
  }
  m_matvar_indexes.resize(nb_item);
  m_local_ids.resize(nb_item);

  // Vérifie qu'on a bien supprimé autant d'entité que prévu.
  Integer nb_remove_computed = (orig_nb_item - nb_item);
  if (nb_remove_computed!=nb_remove)
    ARCANE_FATAL("Bad number of removed material items expected={0} v={1} name={2}",
                 nb_remove,nb_remove_computed,name());
  info(4) << "END_UPDATE_REMOVE nb_removed=" << nb_remove_computed;

  // TODO: il faut recalculer m_max_index_in_multiple_array
  // et compacter éventuellement les variables. (pas indispensable)
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdateRemove(const ComponentModifierWorkInfo& work_info,Integer nb_remove)
{
  Integer nb_item = nbItem();
  Integer orig_nb_item = nb_item;

  //ATTENTION: on modifie nb_item pendant l'itération.
  for (Integer i=0; i<nb_item; ++i) {
    Int32 lid = m_local_ids[i];
    if (work_info.isRemovedCell(lid)) {
      // Déplace le dernier MatVarIndex vers l'élément courant.
      Int32 last = nb_item - 1;
      m_matvar_indexes[i] = m_matvar_indexes[last];
      m_local_ids[i] = m_local_ids[last];
      //info() << "REMOVE ITEM lid=" << lid << " i=" << i;
      --nb_item;
      --i; // Il faut refaire l'itération courante.
    }
  }
  m_matvar_indexes.resize(nb_item);
  m_local_ids.resize(nb_item);

  // Vérifie qu'on a bien supprimé autant d'entité que prévu.
  Integer nb_remove_computed = (orig_nb_item - nb_item);
  if (nb_remove_computed!=nb_remove)
    ARCANE_FATAL("Bad number of removed material items expected={0} v={1} name={2}",
                 nb_remove,nb_remove_computed,name());
  info(4) << "END_UPDATE_REMOVE nb_removed=" << nb_remove_computed;

  // TODO: il faut recalculer m_max_index_in_multiple_array
  // et compacter éventuellement les variables. (pas indispensable)
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  this->_changeLocalIdsV2(this,old_to_new_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
_changeLocalIdsV2(MeshMaterialVariableIndexer* var_indexer,Int32ConstArrayView old_to_new_ids)
{
  // Nouvelle version du changement des localId() qui ne modifie pas l'ordre
  // des m_matvar_indexes.

  ITraceMng* tm = var_indexer->traceMng();

  tm->info(4) << "ChangeLocalIdsV2 name=" << var_indexer->name();
  // Il faut recopier le tableau des localId() car il va être modifié.
  UniqueArray<Int32> ids_copy(var_indexer->localIds());
  UniqueArray<MatVarIndex> matvar_indexes_copy(var_indexer->matvarIndexes());

  var_indexer->m_local_ids.clear();
  var_indexer->m_matvar_indexes.clear();

  Integer nb = ids_copy.size();

  tm->info(4) << "-- -- BEGIN_PROCESSING N=" << ids_copy.size();

  for( Integer i=0; i<nb; ++i ){
    Int32 lid = ids_copy[i];
    Int32 new_lid = old_to_new_ids[lid];
    tm->info(5) << "I=" << i << " lid=" << lid << " new_lid=" << new_lid;

    if (new_lid!=NULL_ITEM_LOCAL_ID){
      MatVarIndex mvi = matvar_indexes_copy[i];
      tm->info(5) << "I=" << i << " new_lid=" << new_lid << " mv=" << mvi;
      Int32 value_index = mvi.valueIndex();
      if (mvi.arrayIndex()==0){
        // TODO: Vérifier si value_index, qui contient le localId() de l'entité
        // ne dois pas être modifié.
        // Normalement, il faudra avoir:
        //    value_index = new_lid;
        // Mais cela plante actuellement car comme on ne récupère pas
        // l'évènement executeReduce() sur les groupes il est possible
        // que nous n'ayons pas les bons ids. (C'est quand même bizarre...)

        // Variable globale: met à jour le localId() dans le MatVarIndex.
        var_indexer->m_matvar_indexes.add(MatVarIndex(0,value_index));
        var_indexer->m_local_ids.add(value_index);
      }
      else{
        // Valeur partielle: rien ne change dans le MatVarIndex
        var_indexer->m_matvar_indexes.add(mvi);
        var_indexer->m_local_ids.add(new_lid);
      }
    }
  }

  // TODO: remplir la fin des tableaux avec des valeurs invalides (pour détecter les problèmes)
  tm->info(4) << "-- -- ChangeLocalIdsV2 END_PROCESSING (V4)"
              << " indexer_name=" << var_indexer->name()
              << " nb_ids=" << var_indexer->m_local_ids.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
transformCells(Int32ConstArrayView nb_env_per_cell,
                Int32ConstArrayView nb_mat_per_cell,
                Int32Array& pure_local_ids,Int32Array& partial_indexes,
                bool is_add_operation,bool is_env, bool is_verbose)
{
  if (is_add_operation)
    _transformPureToPartial(nb_env_per_cell,nb_mat_per_cell,
                            pure_local_ids,partial_indexes,
                            is_env,is_verbose);
  else
    _transformPartialToPure(nb_env_per_cell,nb_mat_per_cell,
                            pure_local_ids,partial_indexes,
                            is_env,is_verbose);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
_transformPureToPartial(Int32ConstArrayView nb_env_per_cell,
                        Int32ConstArrayView nb_mat_per_cell,
                        Int32Array& pure_local_ids,Int32Array& partial_indexes,
                        bool is_env, bool is_verbose)
{
  Integer nb = nbItem();
  for( Integer i=0; i<nb; ++i ){
    MatVarIndex mvi = m_matvar_indexes[i];
    if (mvi.arrayIndex()!=0)
      continue;
    // Comme la maille est pure, le localId() est dans \a mvi
    Int32 local_id = mvi.valueIndex();
    bool do_transform = false;
    // Teste si on doit transformer la maille.
    // Pour un milieu, c'est le cas s'il y a plusieurs milieux.
    // Pour un matériau, c'est le cas s'il y a plusieurs matériaux dans le milieu
    // ou s'il y a plusieurs milieux.
    if (is_env)
      do_transform = nb_env_per_cell[local_id]>1;
    else
      do_transform = nb_env_per_cell[local_id]>1 || nb_mat_per_cell[local_id]>1;
    if (do_transform){
      pure_local_ids.add(local_id);
      Int32 current_index = m_max_index_in_multiple_array+1;
      partial_indexes.add(current_index);
      //TODO: regarder s'il faut faire +1 ou -1 pour m_max_index_in_multiple_array
      //TODO: prendre le premier var_index libre une fois que la liste des index libres existera
      m_matvar_indexes[i] = MatVarIndex(m_index+1,current_index);
      ++m_max_index_in_multiple_array;
      if (is_verbose)
        info() << "Transform pure cell to partial cell i=" << i
               << " local_id=" << mvi.valueIndex()
               << " var_index =" << m_matvar_indexes[i].valueIndex();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
_transformPartialToPure(Int32ConstArrayView nb_env_per_cell,
                        Int32ConstArrayView nb_mat_per_cell,
                        Int32Array& pure_local_ids,Int32Array& partial_indexes,
                        bool is_env, bool is_verbose)
{
  Integer nb = nbItem();
  for( Integer i=0; i<nb; ++i ){
    MatVarIndex mvi = m_matvar_indexes[i];
    if (mvi.arrayIndex()==0)
      continue;
    Int32 local_id = m_local_ids[i];
    Int32 var_index = mvi.valueIndex();
    bool do_transform = false;
    // Teste si on transforme la maille partielle en maille pure.
    // Pour un milieu, c'est le cas s'il n'y a plus qu'un milieu.
    // Pour un matériau, c'est le cas s'il y a plus qu'un matériau et un milieu.
    if (is_env)
      do_transform = nb_env_per_cell[local_id]==1;
    else
      do_transform = nb_env_per_cell[local_id]==1 && nb_mat_per_cell[local_id]==1;
    if (do_transform){
      pure_local_ids.add(local_id);
      partial_indexes.add(var_index);

      m_matvar_indexes[i] = MatVarIndex(0,local_id);

      // TODO: ajouter le var_index à la liste des index libres.

      if (is_verbose)
        info() << "Transform partial cell to pure cell i=" << i
               << " local_id=" << local_id
               << " var_index =" << var_index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
transformCellsV2(ComponentModifierWorkInfo& work_info)
{
  if (work_info.is_add)
    _transformPureToPartialV2(work_info);
  else
    _transformPartialToPureV2(work_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
_transformPureToPartialV2(ComponentModifierWorkInfo& work_info)
{
  Int32Array& pure_local_ids = work_info.pure_local_ids;
  Int32Array& partial_indexes = work_info.partial_indexes;
  bool is_verbose = work_info.is_verbose;

  Integer nb = nbItem();
  for( Integer i=0; i<nb; ++i ){
    MatVarIndex mvi = m_matvar_indexes[i];
    if (mvi.arrayIndex()!=0)
      continue;
    // Comme la maille est pure, le localId() est dans \a mvi
    Int32 local_id = mvi.valueIndex();
    bool do_transform = work_info.isTransformedCell(CellLocalId(local_id));
    if (do_transform){
      pure_local_ids.add(local_id);
      Int32 current_index = m_max_index_in_multiple_array+1;
      partial_indexes.add(current_index);
      //TODO: regarder s'il faut faire +1 ou -1 pour m_max_index_in_multiple_array
      //TODO: prendre le premier var_index libre une fois que la liste des index libres existera
      m_matvar_indexes[i] = MatVarIndex(m_index+1,current_index);
      ++m_max_index_in_multiple_array;
      if (is_verbose)
        info() << "Transform pure cell to partial cell i=" << i
               << " local_id=" << mvi.valueIndex()
               << " var_index =" << m_matvar_indexes[i].valueIndex();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
_transformPartialToPureV2(ComponentModifierWorkInfo& work_info)
{
  Int32Array& pure_local_ids = work_info.pure_local_ids;
  Int32Array& partial_indexes = work_info.partial_indexes;
  bool is_verbose = work_info.is_verbose;

  Integer nb = nbItem();
  for( Integer i=0; i<nb; ++i ){
    MatVarIndex mvi = m_matvar_indexes[i];
    if (mvi.arrayIndex()==0)
      continue;
    Int32 local_id = m_local_ids[i];
    Int32 var_index = mvi.valueIndex();
    bool do_transform = work_info.isTransformedCell(CellLocalId(local_id));
    // Teste si on transforme la maille partielle en maille pure.
    // Pour un milieu, c'est le cas s'il n'y a plus qu'un milieu.
    // Pour un matériau, c'est le cas s'il y a plus qu'un matériau et un milieu.
    if (do_transform){
      pure_local_ids.add(local_id);
      partial_indexes.add(var_index);

      m_matvar_indexes[i] = MatVarIndex(0,local_id);

      // TODO: ajouter le var_index à la liste des index libres.

      if (is_verbose)
        info() << "Transform partial cell to pure cell i=" << i
               << " local_id=" << local_id
               << " var_index =" << var_index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
checkValid()
{
  ValueChecker vc(A_FUNCINFO);

  Integer nb_item = nbItem();

  vc.areEqual(nb_item,m_matvar_indexes.size(),"Incoherent size for local ids and matvar indexes");

  // TODO: vérifier que les m_local_ids pour les parties pures correspondent
  // au m_matvar_indexes.valueIndex() correspondant.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
