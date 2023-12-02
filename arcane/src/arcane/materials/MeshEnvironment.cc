// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironment.cc                                          (C) 2000-2023 */
/*                                                                           */
/* Milieu d'un maillage.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FunctorUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/ItemGroupObserver.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentItemVectorView.h"
#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/MeshComponentPartData.h"
#include "arcane/materials/ComponentPartItemVectorView.h"

#include "arcane/materials/internal/MeshEnvironment.h"
#include "arcane/materials/internal/MeshMaterial.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"
#include "arcane/materials/internal/ComponentItemInternalData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshEnvironmentObserver
: public TraceAccessor
, public IItemGroupObserver
{
 public:
  MeshEnvironmentObserver(MeshEnvironment* env,ITraceMng* tm)
  : TraceAccessor(tm), m_environment(env){}
 public:
  void executeExtend(const Int32ConstArrayView* info1) override
  {
    if (info1){
      info(4) << "EXTEND_ENV " << m_environment->name() << " ids=" << (*info1);
      if (m_environment->materialMng()->isInMeshMaterialExchange())
        info() << "EXTEND_ENV_IN_LOADBALANCE " << m_environment->name()
               << " ids=" << (*info1);
    }
  }
  void executeReduce(const Int32ConstArrayView* info1) override
  {
    if (info1){
      info(4) << "REDUCE_ENV " << m_environment->name() << " ids=" << (*info1);
      if (m_environment->materialMng()->isInMeshMaterialExchange())
        info() << "REDUCE_ENV_IN_LOADBALANCE " << m_environment->name()
               << " ids=" << (*info1);
    }
  }
  void executeCompact(const Int32ConstArrayView* info1) override
  {
    info(4) << "COMPACT_ENV " << m_environment->name();
    if (!info1)
      ARCANE_FATAL("No info available");
    Int32ConstArrayView old_to_new_ids(*info1);
    m_environment->notifyLocalIdsChanged(old_to_new_ids);
  }
  void executeInvalidate() override
  {
    info() << "WARNING: invalidate() is invalid on an environment group! partial values may be corrupted"
           << " env=" << m_environment->name();
  }
  bool needInfo() const override { return true; }
 private:
  MeshEnvironment* m_environment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshEnvironment::
MeshEnvironment(IMeshMaterialMng* mm, const String& name, Int16 env_id)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_nb_mat_per_cell(VariableBuildInfo(mm->mesh(), mm->name() + "_CellNbMaterial_" + name))
, m_data(this, name, env_id, false)
, m_non_const_this(this)
, m_internal_api(this)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshEnvironment::
~MeshEnvironment()
{
  // A priori, pas besoin de détruire l'observer car le groupe associé
  // le fera lorsqu'il sera détruit.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
build()
{
  IMesh* mesh = m_material_mng->mesh();
  IItemFamily* cell_family = mesh->cellFamily();
  String group_name = m_material_mng->name() + "_" + name();
  CellGroup cells = cell_family->findGroup(group_name,true);

  if (m_material_mng->isMeshModificationNotified()){
    m_group_observer = new MeshEnvironmentObserver(this,traceMng());
    cells.internal()->attachObserver(this,m_group_observer);
  }

  m_data._setItems(cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
addMaterial(MeshMaterial* mm)
{
  m_materials.add(mm);
  m_true_materials.add(mm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
setVariableIndexer(MeshMaterialVariableIndexer* idx)
{
  m_data._setVariableIndexer(idx);
  idx->setCells(m_data.items());
  idx->setIsEnvironment(true);

  // S'il n'y qu'un matériau, le variable indexer de ce matériau est
  // aussi 'idx' mais avec un autre groupe associé. Pour que tout soit
  // cohérent, il faut être sur que ce matériau a aussi le même groupe.
  // TODO: pour garantir la cohérence, il faudrait supprimer
  // dans m_data le groupe d'entité.
  if (m_true_materials.size()==1)
    m_true_materials[0]->componentData()->_setItems(m_data.items());
  m_data._buildPartData();
  for( MeshMaterial* mat : m_true_materials )
    mat->componentData()->_buildPartData();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
computeNbMatPerCell()
{
  info(4) << "ComputeNbMatPerCell env=" << name();
  Integer nb_mat = m_materials.size();
  m_nb_mat_per_cell.fill(0);
  Integer total = 0;
  for( Integer i=0; i<nb_mat; ++i ){
    IMeshMaterial* mat = m_materials[i];
    CellGroup mat_cells = mat->cells();
    total += mat_cells.size();
    ENUMERATE_CELL(icell,mat_cells){
      ++m_nb_mat_per_cell[icell];
    }
  }
  m_total_nb_cell_mat = total;
  ENUMERATE_CELL(icell,this->cells()){
    Int32 n = m_nb_mat_per_cell[icell];
    if (n==0)
      ARCANE_FATAL("No material in cell '{0}' for environment '{1}'",
                   icell->uniqueId(),name());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul les infos sur les matériaux.
 *
 * Cette méthode est appelée par le MeshMaterialMng et doit être appelée
 * une fois que les m_items_internal ont été mis à jour et
 * computeNbMatPerCell() et computeItemListForMaterials() ont été appelées
 */
void MeshEnvironment::
computeMaterialIndexes(ComponentItemInternalData* item_internal_data)
{
  info(4) << "Compute (V2) indexes for environment name=" << name();

  IItemFamily* cell_family = cells().itemFamily();
  Integer max_local_id = cell_family->maxLocalId();
  Integer total_nb_cell_mat = m_total_nb_cell_mat;
  item_internal_data->resizeNbMatCellForEnvironment(id(),total_nb_cell_mat);
  ArrayView<ComponentItemInternal> mat_items_internal = item_internal_data->matItemsInternal(id());
  for( Integer i=0; i<total_nb_cell_mat; ++i ){
    ComponentItemInternal& ref_ii = mat_items_internal[i];
    ref_ii._reset();
  } 

  Int32UniqueArray cells_index(max_local_id);
  Int32UniqueArray cells_pos(max_local_id);
  //TODO: regarder comment supprimer ce tableau cells_env qui n'est normalement pas utile
  // car on doit pouvoir directement utiliser les m_items_internal
  UniqueArray<ComponentItemInternal*> cells_env(max_local_id);
  Int32ArrayView nb_mat_per_cell = m_nb_mat_per_cell.asArray();
  ConstArrayView<ComponentItemInternal*> items_internal = itemsInternalView();

  {
    Integer cell_index = 0;
    Int32 env_id = this->id();
    Int32ConstArrayView local_ids = variableIndexer()->localIds();

    for( Integer z=0, nb=local_ids.size(); z<nb; ++z ){
      Int32 lid = local_ids[z];
      Int32 nb_mat = nb_mat_per_cell[lid];
      ComponentItemInternal* env_item = items_internal[z];
      cells_index[lid] = cell_index;
      cells_pos[lid] = cell_index;
      //info(4) << "XZ=" << z << " LID=" << lid << " POS=" << cell_index;
      env_item->_setNbSubItem(nb_mat);
      env_item->_setComponent(this,env_id);
      if (nb_mat!=0){
        env_item->_setFirstSubItem(&mat_items_internal[cell_index]);
        cells_env[lid] = env_item;
      }
      cell_index += nb_mat;
    }
  }
  {
    Integer nb_mat = m_true_materials.size();
    ItemInfoListView items_internal(cell_family->itemInfoListView());
    for( Integer i=0; i<nb_mat; ++i ){
      MeshMaterial* mat = m_true_materials[i];
      Int32 mat_id = mat->id();
      const MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
      CellGroup mat_cells = mat->cells();
      info(4) << "COMPUTE (V2) mat_cells mat=" << mat->name() << " nb_cell=" << mat_cells.size()
              << " mat_id=" << mat_id << " index=" << var_indexer->index();

      mat->resizeItemsInternal(var_indexer->nbItem());

      ConstArrayView<MatVarIndex> matvar_indexes = var_indexer->matvarIndexes();

      ArrayView<ComponentItemInternal*> mat_items_internal_pointer = mat->itemsInternalView();
      Int32ConstArrayView local_ids = var_indexer->localIds();

      for( Integer z=0, nb_id = matvar_indexes.size(); z<nb_id; ++z){
        MatVarIndex mvi = matvar_indexes[z];
        Int32 lid = local_ids[z];
        Int32 pos = cells_pos[lid];
        //info(4) << "Z=" << z << " LID=" << lid << " POS=" << pos;
        ++cells_pos[lid];
        ComponentItemInternal& ref_ii = mat_items_internal[pos];
        mat_items_internal_pointer[z] = &mat_items_internal[pos];
        ref_ii._setSuperAndGlobalItem(cells_env[lid],items_internal[lid]);
        ref_ii._setComponent(mat,mat_id);
        ref_ii._setVariableIndex(mvi);
        ref_ii._setLevel(LEVEL_MATERIAL);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul pour les mailles des matériaux du milieu leur emplacement
 * dans le tableau d'indexation des variables.
 */
void MeshEnvironment::
computeItemListForMaterials(const VariableCellInt32& nb_env_per_cell)
{
  info(4) << "ComputeItemListForMaterials (V2)";

  // Calcul pour chaque matériau le nombre de mailles mixtes
  // TODO: a faire dans MeshMaterialVariableIndexer
  for( MeshMaterial* mat : m_true_materials ){
    MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
    CellGroup cells = var_indexer->cells();
    Integer var_nb_cell = cells.size();

    ComponentItemListBuilder list_builder(var_indexer,0);

    info(4) << "MAT_INDEXER mat=" << mat->name() << " NB_CELL=" << var_nb_cell << " name=" << cells.name();
    ENUMERATE_CELL(icell,cells){
      Int32 lid = icell.itemLocalId();
      // On ne prend l'indice global que si on est le seul matériau et le seul
      // milieu de la maille. Sinon, on prend un indice multiple
      if (nb_env_per_cell[icell]>1 || m_nb_mat_per_cell[icell]>1)
        list_builder.addPartialItem(lid);
      else
        list_builder.addPureItem(lid);
    }

    if (traceMng()->verbosityLevel()>=5)
      info() << "MAT_NB_MULTIPLE_CELL (V2) mat=" << var_indexer->name()
             << " nb_in_global=" << list_builder.pureMatVarIndexes().size()
             << " (ids=" << list_builder.pureMatVarIndexes() << ")"
             << " nb_in_multiple=" << list_builder.partialMatVarIndexes().size()
             << " (ids=" << list_builder.partialLocalIds() << ")";
    var_indexer->endUpdate(list_builder);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
_addItemsToIndexer(const VariableCellInt32& nb_env_per_cell,
                   MeshMaterialVariableIndexer* var_indexer,
                   Int32ConstArrayView local_ids)
{
  ComponentItemListBuilder list_builder(var_indexer,var_indexer->maxIndexInMultipleArray());

  for( Int32 lid : local_ids ){
    CellLocalId cell_id(lid);
    // On ne prend l'indice global que si on est le seul matériau et le seul
    // milieu de la maille. Sinon, on prend un indice multiple
    if (nb_env_per_cell[cell_id]>1 || m_nb_mat_per_cell[cell_id]>1)
      list_builder.addPartialItem(lid);
    else
      list_builder.addPureItem(lid);
  }

  if (traceMng()->verbosityLevel()>=5)
    info() << "ADD_MATITEM_TO_INDEXER component=" << var_indexer->name()
           << " nb_pure=" << list_builder.pureMatVarIndexes().size()
           << " nb_partial=" << list_builder.partialMatVarIndexes().size()
           << "\n pure=(" << list_builder.pureMatVarIndexes() << ")"
           << "\n partial=(" << list_builder.partialMatVarIndexes() << ")";

  var_indexer->endUpdateAdd(list_builder);

  // Maintenant que les nouveaux MatVar sont créés, il faut les
  // initialiser avec les bonnes valeurs.
  functor::apply(m_material_mng,&IMeshMaterialMng::visitVariables,
                 [&](IMeshMaterialVariable* mv) { mv->_internalApi()->initializeNewItems(list_builder); }
                 );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute les mailles d'un matériau du milieu.
 *
 * Ajoute les mailles données par \a local_ids au matériau \a mat
 * du milieu. L'indexeur du matériau est mis à jour et si \a update_env_indexer
 * est vrai, celui du milieu aussi (ce qui signifie que le milieu apparait
 * dans les mailles \a local_ids).
 */
void MeshEnvironment::
_addItemsDirect(const VariableCellInt32& nb_env_per_cell,MeshMaterial* mat,
                Int32ConstArrayView local_ids,bool update_env_indexer)
{
  info(4) << "MeshEnvironment::addItemsDirect" << " mat=" << mat->name();

  MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
  Int32 nb_to_add = local_ids.size();

  // Met à jour le nombre de matériaux par maille et le nombre total de mailles matériaux.
  for( Int32 lid : local_ids )
    ++m_nb_mat_per_cell[CellLocalId{lid}];

  m_total_nb_cell_mat += nb_to_add;

  _addItemsToIndexer(nb_env_per_cell,var_indexer,local_ids);

  if (update_env_indexer){
    // Met aussi à jour les entités \a local_ids à l'indexeur du milieu.
    // Cela n'est possible que si le nombre de matériaux du milieu
    // est supérieur ou égal à 2 (car sinon le matériau et le milieu
    // ont le même indexeur)
    _addItemsToIndexer(nb_env_per_cell,this->variableIndexer(),local_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Supprime les mailles d'un matériau du milieu.
 *
 * Supprime les mailles données par \a local_ids du matériau \a mat
 * du milieu. L'indexeur du matériau est mis à jour et si \a update_env_indexer
 * est vrai, celui du milieu aussi (ce qui signifie que le milieu disparait
 * des mailles \a local_ids).
 *
 * TODO: optimiser cela en ne parcourant pas toutes les mailles
 * matériaux du milieu (il faut supprimer removed_local_ids_filter).
 * Si on connait l'indice de chaque maille dans la liste des MatVarIndex
 * de l'indexeur, on peut directement taper dedans.
 */
void MeshEnvironment::
_removeItemsDirect(MeshMaterial* mat,Int32ConstArrayView local_ids,
                   bool update_env_indexer)
{
  info(4) << "MeshEnvironment::removeItemsDirect mat=" << mat->name();

  IItemFamily* cell_family = cells().itemFamily();

  Integer nb_to_remove = local_ids.size();

  UniqueArray<bool> removed_local_ids_filter(cell_family->maxLocalId(),false);

  // Met à jour le nombre de matériaux par maille et le nombre total de mailles matériaux.
  for( Integer i=0; i<nb_to_remove; ++i ){
    Int32 lid = local_ids[i];
    CellLocalId cell_lid(lid);
    --m_nb_mat_per_cell[cell_lid];
    removed_local_ids_filter[lid] = true;
  }
  m_total_nb_cell_mat -= nb_to_remove;

  mat->variableIndexer()->endUpdateRemove(removed_local_ids_filter,nb_to_remove);

  if (update_env_indexer){
    // Met aussi à jour les entités \a local_ids à l'indexeur du milieu.
    // Cela n'est possible que si le nombre de matériaux du milieu
    // est supérieur ou égal à 2 (car sinon le matériau et le milieu
    // ont le même indexeur)
    this->variableIndexer()->endUpdateRemove(removed_local_ids_filter,nb_to_remove);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
updateItemsDirect(const VariableCellInt32& nb_env_per_cell,MeshMaterial* mat,
                  Int32ConstArrayView local_ids,bool is_add_operation,
                  bool update_env_indexer)
{
  if (is_add_operation)
    _addItemsDirect(nb_env_per_cell,mat,local_ids,update_env_indexer);
  else
    _removeItemsDirect(mat,local_ids,update_env_indexer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
notifyLocalIdsChanged(Int32ConstArrayView old_to_new_ids)
{
  // NOTE:
  // Cette méthode est appelée lorsqu'il y a un compactage du maillage
  // et le groupe d'entité associé à ce milieu vient d'être compacté.
  // Comme actuellement il n'y a pas d'observeurs pour l'ajout
  // ou la suppression de mailles du groupe, il est possible
  // lorsque cette méthode est appelée que les infos des milieux et
  // matériaux ne soient pas à jour (par exemple, la liste des local_ids
  // du m_variable_indexer n'a pas les mêmes valeurs que cells().
  // Pour l'instant ce n'est pas très grave car tout est écrasé après
  // chaque modif sur un matériau ou un milieu.
  // A terme, il faudra prendre cela en compte lorsque l'ajout
  // où la suppression de mailles matériaux sera optimisée.
  info(4) << "Changing (V3) local ids references env=" << name();
  info(4) << "CurrentCells name=" << cells().name()
          << " n=" << cells().view().localIds().size();
  info(4) << "MatVarIndex name=" << cells().name()
          << " n=" << variableIndexer()->matvarIndexes().size();
  Integer nb_mat = m_true_materials.size();
  info(4) << "NotifyLocalIdsChanged env=" << name() << " nb_mat=" << nb_mat
         << " old_to_new_ids.size=" << old_to_new_ids.size();

  // Si le milieu n'a qu'un seul matériau, ils partagent le même variable_indexer
  // donc il ne faut changer les ids qu'une seule fois. Par contre, le
  // tableau m_items_internal n'est pas partagé entre le matériau
  // et le milieu donc il faut recalculer les infos séparément.
  // Il faut le faire pour le milieu avant de mettre à jour les infos du matériau car
  // une fois ceci fait la valeur m_variable_indexer->m_local_ids_in_indexes_view
  // aura changé et il ne sera plus possible de déterminer la correspondance
  // entre les nouveaux et les anciens localId

  if (nb_mat==1){
    m_data._changeLocalIdsForInternalList(old_to_new_ids);
    MeshMaterial* true_mat = m_true_materials[0];
    _changeIds(true_mat->componentData(),old_to_new_ids);
  }
  else{
    // Change les infos des matériaux
    for( Integer i=0; i<nb_mat; ++i ){
      MeshMaterial* true_mat = m_true_materials[i];
      info(4) << "ChangeIds MAT i=" << i << " MAT=" << true_mat->name();
      _changeIds(true_mat->componentData(),old_to_new_ids);
    }
    // Change les infos du milieu
    _changeIds(componentData(),old_to_new_ids);
  }

  // Reconstruit les infos sur les mailles pures et mixtes.
  // Il faut le faire une fois que tous les valeurs sont à jour.
  {
    for( Integer i=0; i<nb_mat; ++i ){
      MeshMaterial* true_mat = m_true_materials[i];
      true_mat->componentData()->_rebuildPartData();
    }
    componentData()->_rebuildPartData();
  }

  checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
_changeIds(MeshComponentData* cdata,Int32ConstArrayView old_to_new_ids)
{
  info(4) << "ChangeIds() (V4) for name=" << cdata->name();
  info(4) << "Use new version for ChangeIds()";

  cdata->_changeLocalIdsForInternalList(old_to_new_ids);
  cdata->variableIndexer()->changeLocalIds(old_to_new_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCell MeshEnvironment::
findEnvCell(AllEnvCell c) const
{
  Int32 env_id = m_data.componentId();
  ENUMERATE_CELL_ENVCELL(ienvcell,c){
    EnvCell ec = *ienvcell;
    Int32 eid = ec.environmentId();
    if (eid==env_id)
      return ec;
  }
  return EnvCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentCell MeshEnvironment::
findComponentCell(AllEnvCell c) const
{
  return findEnvCell(c);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvItemVectorView MeshEnvironment::
envView() const
{
  return { m_non_const_this, variableIndexer()->matvarIndexes(),
           itemsInternalView(), variableIndexer()->localIds() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView MeshEnvironment::
view() const
{
  return envView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
resizeItemsInternal(Integer nb_item)
{
  m_data._resizeItemsInternal(nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView MeshEnvironment::
pureItems() const
{
  return m_data._partData()->pureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView MeshEnvironment::
impureItems() const
{
  return m_data._partData()->impureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPartItemVectorView MeshEnvironment::
partItems(eMatPart part) const
{
  return m_data._partData()->partView(part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvPurePartItemVectorView MeshEnvironment::
pureEnvItems() const
{
  return { m_non_const_this, m_data._partData()->pureView() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvImpurePartItemVectorView MeshEnvironment::
impureEnvItems() const
{
  return { m_non_const_this, m_data._partData()->impureView() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvPartItemVectorView MeshEnvironment::
partEnvItems(eMatPart part) const
{
  return { m_non_const_this, m_data._partData()->partView(part) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
checkValid()
{
  if (!arcaneIsCheck())
    return;

  m_data.checkValid();

  for( IMeshMaterial* mat : m_materials ){
    mat->checkValid();
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MeshEnvironment::InternalApi::
variableIndexerIndex() const
{
  return variableIndexer()->index();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
