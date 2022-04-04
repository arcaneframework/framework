﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentData.cc                                        (C) 2000-2017 */
/*                                                                           */
/* Données d'un constituant (matériau ou milieu) d'un maillage.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"

#include "arcane/materials/MeshComponentPartData.h"
#include "arcane/materials/MeshComponentData.h"
#include "arcane/materials/MeshMaterialVariableIndexer.h"
#include "arcane/materials/IMeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshComponentData::
MeshComponentData(IMeshComponent* component,const String& name,
                  Int32 component_id,bool create_indexer)
: TraceAccessor(component->traceMng())
, m_component(component)
, m_component_id(component_id)
, m_name(name)
, m_is_indexer_owner(false)
, m_variable_indexer(nullptr)
, m_part_data(nullptr)
{
  if (create_indexer){
    m_variable_indexer = new MeshMaterialVariableIndexer(traceMng(),name);
    m_is_indexer_owner = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshComponentData::
~MeshComponentData()
{
  delete m_part_data;
  if (m_is_indexer_owner)
    delete m_variable_indexer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
resizeItemsInternal(Integer nb_item)
{
  m_items_internal.resize(nb_item);
  if (m_part_data)
    m_part_data->setComponentItemInternalView(m_items_internal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
setVariableIndexer(MeshMaterialVariableIndexer* indexer)
{
  m_variable_indexer = indexer;
  m_is_indexer_owner = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
setItems(const ItemGroup& group)
{
  m_items = group;
  if (m_variable_indexer)
    m_variable_indexer->setCells(group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
buildPartData()
{
  m_part_data = new MeshComponentPartData(m_component);
  m_part_data->setComponentItemInternalView(m_items_internal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Met à jour les m_items_internal du constituant
 * après changement de la numérotation locale.
 *
 * \warning il faut appeler cette méthode avant de mettre à jour le
 * m_variable_indexer car on se sert de ses local_ids.
 */
void MeshComponentData::
changeLocalIdsForInternalList(Int32ConstArrayView old_to_new_ids)
{
  ItemInternalList global_item_list = items().itemFamily()->itemsInternal();

  // TODO: regarder s'il est possible de supprimer le tableau temporaire
  // new_internals (c'est à peu près sur que c'est possible).
  ConstArrayView<ComponentItemInternal*> current_internals(itemsInternalView());
  UniqueArray<ComponentItemInternal*> new_internals;

  Int32ConstArrayView local_ids = variableIndexer()->localIds();

  for( Integer i=0, nb=current_internals.size(); i<nb; ++i ){
    Int32 lid = local_ids[i];
    Int32 new_lid = old_to_new_ids[lid];
    if (new_lid!=NULL_ITEM_LOCAL_ID){
      new_internals.add(current_internals[i]);
      current_internals[i]->setGlobalItem(global_item_list[new_lid]);
#if MAT_UPDATE_LOCALID
      MatVarIndex mvi = current_internals[i]->variableIndex();
      if (mvi.arrayIndex()==0){
        mat_type->info() << "NEW_VALUE_INDEX i=" << i << " old=" << mvi.valueIndex()
                         << " new=" << old_to_new_ids[mvi.valueIndex()]
                         << " X=" << current_internals[i];
        current_internals[i]->setVariableIndex(MatVarIndex(0,old_to_new_ids[mvi.valueIndex()]));
      }
#endif
    }
  }

  // TODO: regarder supprimer cette copie aussi.
  {
    resizeItemsInternal(new_internals.size());
    itemsInternalView().copy(new_internals);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
rebuildPartData()
{
  if (!m_part_data)
    buildPartData();
  m_part_data->setComponentItemInternalView(m_items_internal);
  m_part_data->setFromMatVarIndexes(m_variable_indexer->matvarIndexes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
checkValid()
{
  if (!arcaneIsCheck())
    return;

  info(4) << "Check valid component name=" << name();
  m_variable_indexer->checkValid();

  // Vérifie que les groupes sont cohérents entre cette instance
  // et le groupe dans m_variable_indexer
  if (m_items!=m_variable_indexer->cells())
    ARCANE_FATAL("Incoherent group for component name={0} data={1} indexer={2}",
                 name(),m_items.name(),m_variable_indexer->cells().name());

  // Vérifie que la liste des indexer->localIds() et celle
  // du groupe 'cells' est la meme. Pour cela, on déclare un tableau
  // qu'on indexe par le localId() de la maille. Pour chaque élément on
  // ajoute 1 si la maille est dans le groupe et 2 si elle est dans les
  // structures internes du composant. Si la valeur finale n'est pas 3 il
  // y a incohérence.
  {
    IItemFamily* family = m_items.itemFamily();
    UniqueArray<Int32> presence(family->maxLocalId());
    presence.fill(0);
    ENUMERATE_ITEM(iitem,m_items){
      presence[iitem.itemLocalId()] = 1;
    }
    Int32ConstArrayView indexer_local_ids = m_variable_indexer->localIds();
    for( Integer i=0, n=indexer_local_ids.size(); i<n; ++i )
      presence[indexer_local_ids[i]] += 2;
    ItemInternalList items_internal = family->itemsInternal();
    Integer nb_error = 0;
    for( Integer i=0, n=presence.size(); i<n; ++i ){
      Int32 v = presence[i];
      if (v==3 || v==0)
        continue;
      Cell cell(items_internal[i]);
      ++nb_error;
      info(4) << "WARNING: Incoherence between group and internals "
              << " component=" << name() << " v=" << v
              << " cell=" << ItemPrinter(cell);
    }
    if (nb_error!=0){
      warning() << "WARNING: Incoherence between group and internals "
                << " component=" << name() << " nb_error=" << nb_error;
    }
  }

  // Vérifie la cohérence des MatVarIndex entre le tableau direct
  // et les valeurs dans les m_items_internal
  ConstArrayView<MatVarIndex> mat_var_indexes(m_variable_indexer->matvarIndexes());
  Integer nb_val = mat_var_indexes.size();
  info(4) << "CheckValid component_name=" << name()
         << " matvar_indexes=" << mat_var_indexes;
  info(4) << "Cells=" << m_variable_indexer->cells().view().localIds();
  for( Integer i=0; i<nb_val; ++ i){
    MatVarIndex component_mvi = m_items_internal[i]->variableIndex();
    MatVarIndex mvi = mat_var_indexes[i];
    if (component_mvi!=mvi)
      ARCANE_FATAL("Bad 'var_index' environment={3} component='{0}' direct='{1}' i={2}",
                   component_mvi,mvi,i,name());
  }

  if (m_part_data)
    m_part_data->checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
