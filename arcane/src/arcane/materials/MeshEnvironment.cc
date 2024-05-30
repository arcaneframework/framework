// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironment.cc                                          (C) 2000-2024 */
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
#include "arcane/utils/ArraySimdPadder.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/ItemGroupObserver.h"
#include "arcane/core/internal/ItemGroupImplInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentItemVectorView.h"
#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/ComponentPartItemVectorView.h"

#include "arcane/materials/internal/MeshEnvironment.h"
#include "arcane/materials/internal/MeshMaterial.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"
#include "arcane/materials/internal/ComponentItemInternalData.h"
#include "arcane/materials/internal/ConstituentConnectivityList.h"
#include "arcane/materials/internal/ConstituentItemVectorImpl.h"
#include "arcane/materials/internal/MeshComponentPartData.h"

#include "arcane/accelerator/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Scan.h"
#include "arcane/accelerator/SpanViews.h"

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

  MeshEnvironmentObserver(MeshEnvironment* env, ITraceMng* tm)
  : TraceAccessor(tm)
  , m_environment(env)
  {}

 public:

  void executeExtend(const Int32ConstArrayView* info1) override
  {
    if (info1) {
      //info(4) << "EXTEND_ENV " << m_environment->name() << " ids=" << (*info1);
      if (m_environment->materialMng()->isInMeshMaterialExchange())
        info() << "EXTEND_ENV_IN_LOADBALANCE " << m_environment->name()
               << " ids=" << (*info1);
    }
  }
  void executeReduce(const Int32ConstArrayView* info1) override
  {
    if (info1) {
      //info(4) << "REDUCE_ENV " << m_environment->name() << " ids=" << (*info1);
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
, m_data(this, name, env_id, mm->_internalApi()->componentItemSharedInfo(LEVEL_ENVIRONMENT), false)
, m_non_const_this(this)
, m_internal_api(this)
{
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
  CellGroup cells = cell_family->findGroup(group_name, true);
  cells._internalApi()->setAsConstituentGroup();

  if (m_material_mng->isMeshModificationNotified()) {
    m_group_observer = new MeshEnvironmentObserver(this, traceMng());
    cells.internal()->attachObserver(this, m_group_observer);
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
  if (m_true_materials.size() == 1)
    m_true_materials[0]->componentData()->_setItems(m_data.items());
  m_data._buildPartData();
  for (MeshMaterial* mat : m_true_materials)
    mat->componentData()->_buildPartData();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
computeNbMatPerCell()
{
  info(4) << "ComputeNbMatPerCell env=" << name();
  Integer nb_mat = m_materials.size();
  Integer total = 0;
  for (Integer i = 0; i < nb_mat; ++i) {
    IMeshMaterial* mat = m_materials[i];
    CellGroup mat_cells = mat->cells();
    total += mat_cells.size();
  }
  m_total_nb_cell_mat = total;
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
computeMaterialIndexes(ComponentItemInternalData* item_internal_data, RunQueue& queue)
{
  info(4) << "Compute (V2) indexes for environment name=" << name();
  const bool is_mono_mat = (nbMaterial() == 1 && (cells().size() == totalNbCellMat()));
  if (is_mono_mat) {
    _computeMaterialIndexesMonoMat(item_internal_data, queue);
  }
  else
    _computeMaterialIndexes(item_internal_data, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
_computeMaterialIndexes(ComponentItemInternalData* item_internal_data, RunQueue& queue)
{
  IItemFamily* cell_family = cells().itemFamily();
  Integer max_local_id = cell_family->maxLocalId();
  const Int16 env_id = componentId();

  ComponentItemInternalRange mat_items_internal_range = item_internal_data->matItemsInternalRange(id());

  UniqueArray<Int32> cells_index(platform::getDefaultDataAllocator());
  cells_index.resize(max_local_id);
  UniqueArray<Int32> cells_pos(platform::getDefaultDataAllocator());
  cells_pos.resize(max_local_id);

  // TODO: regarder comment supprimer ce tableau cells_env qui n'est normalement pas utile
  // car on doit pouvoir directement utiliser les m_items_internal
  UniqueArray<ConstituentItemIndex> cells_env(platform::getDefaultDataAllocator());
  cells_env.resize(max_local_id);

  Int32ConstArrayView local_ids = variableIndexer()->localIds();
  ConstituentItemLocalIdListView constituent_item_list_view = m_data.constituentItemListView();
  const bool do_old = false;
  if (do_old) {
    Integer cell_index = 0;
    for (Integer z = 0, nb = local_ids.size(); z < nb; ++z) {
      Int32 lid = local_ids[z];
      matimpl::ConstituentItemBase env_item = constituent_item_list_view._constituenItemBase(z);
      Int32 nb_mat = env_item.nbSubItem();
      cells_index[lid] = cell_index;
      cell_index += nb_mat;
    }
  }
  else {
    // Calcul l'indice de la première MatCell pour chaque Cell.
    Int32 nb_id = local_ids.size();
    {
      Accelerator::GenericScanner scanner(queue);
      auto cells_index_view = viewOut(queue, cells_index);

      auto getter = [=] ARCCORE_HOST_DEVICE(Int32 index) -> Int32 {
        return constituent_item_list_view._constituenItemBase(index).nbSubItem();
      };
      auto setter = [=] ARCCORE_HOST_DEVICE(Int32 index, Int32 value) {
        Int32 lid = local_ids[index];
        cells_index_view[lid] = value;
      };
      Accelerator::ScannerSumOperator<Int32> op;
      scanner.applyWithIndexExclusive(nb_id, 0, getter, setter, op, A_FUNCINFO);
    }
  }
  {
    auto command = makeCommand(queue);
    auto cells_env_view = viewOut(command, cells_env);
    auto cells_index_view = viewIn(command, cells_index);
    auto cells_pos_view = viewOut(command, cells_pos);
    Int32 nb_id = local_ids.size();

    command << RUNCOMMAND_LOOP1(iter, nb_id)
    {
      auto [z] = iter();
      Int32 lid = local_ids[z];
      matimpl::ConstituentItemBase env_item = constituent_item_list_view._constituenItemBase(z);
      Int32 nb_mat = env_item.nbSubItem();
      Int32 cell_index = cells_index_view[lid];
      cells_pos_view[lid] = cell_index;
      if (nb_mat != 0) {
        env_item._setFirstSubItem(mat_items_internal_range[cell_index]);
      }
      cells_env_view[lid] = env_item.constituentItemIndex();
    };
  }
  {
    Integer nb_mat = m_true_materials.size();
    for (Integer i = 0; i < nb_mat; ++i) {
      MeshMaterial* mat = m_true_materials[i];
      Int16 mat_id = mat->componentId();
      const MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
      CellGroup mat_cells = mat->cells();
      info(4) << "COMPUTE (V2) mat_cells mat=" << mat->name() << " nb_cell=" << mat_cells.size()
              << " mat_id=" << mat_id << " index=" << var_indexer->index();

      mat->resizeItemsInternal(var_indexer->nbItem());

      auto command = makeCommand(queue);
      auto matvar_indexes = viewIn(command, var_indexer->matvarIndexes());
      auto local_ids = viewIn(command, var_indexer->localIds());
      SmallSpan<Int32> cells_pos_view(cells_pos);
      auto cells_env_view = viewIn(command, cells_env);
      ComponentItemSharedInfo* mat_shared_info = item_internal_data->matSharedInfo();
      ComponentItemInternalRange mat_item_internal_range(item_internal_data->matItemsInternalRange(env_id));
      SmallSpan<ConstituentItemIndex> mat_id_list = mat->componentData()->m_constituent_local_id_list.mutableLocalIds();
      const Int32 nb_id = local_ids.size();
      Span<Int32> mat_cells_local_id = mat_cells._internalApi()->itemsLocalId();
      command << RUNCOMMAND_LOOP1(iter, nb_id)
      {
        auto [z] = iter();
        MatVarIndex mvi = matvar_indexes[z];
        Int32 lid = local_ids[z];
        Int32 pos = cells_pos_view[lid];
        ++cells_pos_view[lid];
        ConstituentItemIndex cii = mat_item_internal_range[pos];
        matimpl::ConstituentItemBase ref_ii(mat_shared_info, cii);
        mat_id_list[z] = cii;
        ref_ii._setSuperAndGlobalItem(cells_env_view[lid], ItemLocalId(lid));
        ref_ii._setComponent(mat_id);
        ref_ii._setVariableIndex(mvi);
        // Le rang 0 met à jour le padding SIMD du groupe associé au matériau
        if (z == 0)
          ArraySimdPadder::applySimdPaddingView(mat_cells_local_id);
      };
      mat_cells._internalApi()->notifySimdPaddingDone();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul les infos sur les matériaux en mono-matériaux.
 *
 * Spécialisation pour le cas où le milieu n'a qu'un matériau.
 */
void MeshEnvironment::
_computeMaterialIndexesMonoMat(ComponentItemInternalData* item_internal_data, RunQueue& queue)
{
  const Int16 env_id = componentId();

  ConstituentItemLocalIdListView constituent_item_list_view = m_data.constituentItemListView();

  MeshMaterial* mat = m_true_materials[0];
  const Int16 mat_id = mat->componentId();
  const MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
  CellGroup mat_cells = mat->cells();
  info(4) << "COMPUTE (V2) mat_cells mat=" << mat->name() << " nb_cell=" << mat_cells.size()
          << " mat_id=" << mat_id << " index=" << var_indexer->index();

  mat->resizeItemsInternal(var_indexer->nbItem());

  auto command = makeCommand(queue);
  auto matvar_indexes = viewIn(command, var_indexer->matvarIndexes());
  auto local_ids = viewIn(command, var_indexer->localIds());
  ComponentItemSharedInfo* mat_shared_info = item_internal_data->matSharedInfo();
  ComponentItemInternalRange mat_item_internal_range(item_internal_data->matItemsInternalRange(env_id));
  SmallSpan<ConstituentItemIndex> mat_id_list = mat->componentData()->m_constituent_local_id_list.mutableLocalIds();
  const Int32 nb_id = local_ids.size();
  Span<Int32> mat_cells_local_id = mat_cells._internalApi()->itemsLocalId();
  command << RUNCOMMAND_LOOP1(iter, nb_id)
  {
    auto [z] = iter();
    MatVarIndex mvi = matvar_indexes[z];
    const Int32 lid = local_ids[z];
    const Int32 pos = z;
    matimpl::ConstituentItemBase env_item = constituent_item_list_view._constituenItemBase(z);
    ConstituentItemIndex cii = mat_item_internal_range[pos];
    env_item._setFirstSubItem(cii);

    matimpl::ConstituentItemBase ref_ii(mat_shared_info, cii);
    mat_id_list[z] = cii;
    ref_ii._setSuperAndGlobalItem(env_item.constituentItemIndex(), ItemLocalId(lid));
    ref_ii._setComponent(mat_id);
    ref_ii._setVariableIndex(mvi);
    // Le rang 0 met à jour le padding SIMD du groupe associé au matériau
    if (z == 0)
      ArraySimdPadder::applySimdPaddingView(mat_cells_local_id);
  };
  mat_cells._internalApi()->notifySimdPaddingDone();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul pour les mailles des matériaux du milieu leur emplacement
 * dans le tableau d'indexation des variables.
 */
void MeshEnvironment::
computeItemListForMaterials(const ConstituentConnectivityList& connectivity_list)
{
  info(4) << "ComputeItemListForMaterials (V2)";
  ConstArrayView<Int16> nb_env_per_cell = connectivity_list.cellsNbEnvironment();
  const Int16 env_id = componentId();
  // Calcul pour chaque matériau le nombre de mailles mixtes
  // TODO: a faire dans MeshMaterialVariableIndexer
  for (MeshMaterial* mat : m_true_materials) {
    MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
    CellGroup cells = var_indexer->cells();
    Integer var_nb_cell = cells.size();

    ComponentItemListBuilderOld list_builder(var_indexer, 0);

    info(4) << "MAT_INDEXER mat=" << mat->name() << " NB_CELL=" << var_nb_cell << " name=" << cells.name();
    ENUMERATE_CELL (icell, cells) {
      Int32 lid = icell.itemLocalId();
      // On ne prend l'indice global que si on est le seul matériau et le seul
      // milieu de la maille. Sinon, on prend un indice multiple
      if (nb_env_per_cell[lid] > 1 || connectivity_list.cellNbMaterial(icell, env_id) > 1)
        list_builder.addPartialItem(lid);
      else
        list_builder.addPureItem(lid);
    }

    if (traceMng()->verbosityLevel() >= 5)
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

  if (nb_mat == 1) {
    m_data._changeLocalIdsForInternalList(old_to_new_ids);
    MeshMaterial* true_mat = m_true_materials[0];
    _changeIds(true_mat->componentData(), old_to_new_ids);
  }
  else {
    // Change les infos des matériaux
    for (Integer i = 0; i < nb_mat; ++i) {
      MeshMaterial* true_mat = m_true_materials[i];
      info(4) << "ChangeIds MAT i=" << i << " MAT=" << true_mat->name();
      _changeIds(true_mat->componentData(), old_to_new_ids);
    }
    // Change les infos du milieu
    _changeIds(componentData(), old_to_new_ids);
  }

  // Reconstruit les infos sur les mailles pures et mixtes.
  // Il faut le faire une fois que tous les valeurs sont à jour.
  {
    RunQueue& queue = m_material_mng->_internalApi()->runQueue();
    for (Integer i = 0; i < nb_mat; ++i) {
      MeshMaterial* true_mat = m_true_materials[i];
      true_mat->componentData()->_rebuildPartData(queue);
    }
    componentData()->_rebuildPartData(queue);
  }

  checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironment::
_changeIds(MeshComponentData* cdata, Int32ConstArrayView old_to_new_ids)
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
  ENUMERATE_CELL_ENVCELL (ienvcell, c) {
    EnvCell ec = *ienvcell;
    Int32 eid = ec.environmentId();
    if (eid == env_id)
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
           constituentItemListView(), variableIndexer()->localIds() };
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

  for (IMeshMaterial* mat : m_materials) {
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

Ref<IConstituentItemVectorImpl> MeshEnvironment::InternalApi::
createItemVectorImpl() const
{
  auto* x = new ConstituentItemVectorImpl(m_environment->m_non_const_this);
  return makeRef<IConstituentItemVectorImpl>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IConstituentItemVectorImpl> MeshEnvironment::InternalApi::
createItemVectorImpl(ComponentItemVectorView rhs) const
{
  auto* x = new ConstituentItemVectorImpl(rhs);
  return makeRef<IConstituentItemVectorImpl>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
