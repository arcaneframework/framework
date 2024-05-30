// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironment.h                                           (C) 2000-2024 */
/*                                                                           */
/* Milieu d'un maillage.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHENVIRONMENT_H
#define ARCANE_MATERIALS_INTERNAL_MESHENVIRONMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/MeshVariableScalarRef.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/materials/IMeshEnvironment.h"
#include "arcane/core/materials/ComponentItemInternal.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

#include "arcane/materials/internal/MeshComponentData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class MeshMaterial;
class ComponentItemInternalData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Milieu d'un maillage.
 *
 * Cette classe est à usage interne à Arcane et ne doit pas être utilisée
 * explicitement. Il faut utiliser l'interface IMeshEnvironment pour accéder
 * aux milieux.
 */
class MeshEnvironment
: public TraceAccessor
, public IMeshEnvironment
{
  class InternalApi
  : public IMeshComponentInternal
  {
   public:
    InternalApi(MeshEnvironment* env) : m_environment(env){}
   public:
    MeshMaterialVariableIndexer* variableIndexer() const override
    {
      return m_environment->variableIndexer();
    }
    ConstituentItemLocalIdListView constituentItemListView() const override
    {
      return m_environment->constituentItemListView();
    }
    Int32 variableIndexerIndex() const override;
    Ref<IConstituentItemVectorImpl> createItemVectorImpl() const override;
    Ref<IConstituentItemVectorImpl> createItemVectorImpl(ComponentItemVectorView rhs) const override;

   private:

    MeshEnvironment* m_environment;
  };

 public:

  MeshEnvironment(IMeshMaterialMng* mm,const String& name,Int16 env_id);

 public:

  IMeshMaterialMng* materialMng() override { return m_material_mng; }
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }
  String name() const override { return m_data.name(); }
  CellGroup cells() const override { return m_data.items(); }
  ConstArrayView<IMeshMaterial*> materials() override
  {
    return m_materials;
  }
  Integer nbMaterial() const override
  {
    return m_materials.size();
  }
  MeshMaterialVariableIndexer* variableIndexer() const
  {
    return m_data.variableIndexer();
  }
  ConstituentItemLocalIdListView constituentItemListView() const
  {
    return m_data.constituentItemListView();
  }
  Int32 id() const override
  {
    return m_data.componentId();
  }

  IUserMeshEnvironment* userEnvironment() const override { return m_user_environment; }
  void setUserEnvironment(IUserMeshEnvironment* umm) override { m_user_environment = umm; }

  ComponentCell findComponentCell(AllEnvCell c) const override;
  EnvCell findEnvCell(AllEnvCell c) const override;

  ComponentItemVectorView view() const override;
  EnvItemVectorView envView() const override;

  void checkValid() override;

  bool isMaterial() const override { return false; }
  bool isEnvironment() const override { return true; }
  bool hasSpace(MatVarSpace space) const override
  {
    return space==MatVarSpace::MaterialAndEnvironment || space==MatVarSpace::Environment;
  }
  IMeshMaterial* asMaterial() override { return nullptr; }
  IMeshEnvironment* asEnvironment() override { return this; }

  ComponentPurePartItemVectorView pureItems() const override;
  ComponentImpurePartItemVectorView impureItems() const override;
  ComponentPartItemVectorView partItems(eMatPart part) const override;

  EnvPurePartItemVectorView pureEnvItems() const override;
  EnvImpurePartItemVectorView impureEnvItems() const override;
  EnvPartItemVectorView partEnvItems(eMatPart part) const override;

 public:

  IMeshComponentInternal* _internalApi() override { return &m_internal_api; }

 public:

  void setConstituentItem(Int32 index, ConstituentItemIndex id)
  {
    m_data._setConstituentItem(index,id);
  }
  Int16 componentId() const { return m_data.componentId(); }

 public:

  //! Fonctions publiques mais réservées au IMeshMaterialMng
  //@{
  void build();
  void addMaterial(MeshMaterial* mm);
  void setVariableIndexer(MeshMaterialVariableIndexer* idx);
  //! Recalcule le nombre de mailles par matériau et de mailles totales
  void computeNbMatPerCell();

  void computeItemListForMaterials(const ConstituentConnectivityList& connectivity_list);

  //! Nombre total de mailles pour tous les matériaux
  Integer totalNbCellMat() const { return m_total_nb_cell_mat; }
  void addToTotalNbCellMat(Int32 v) { m_total_nb_cell_mat += v; }

  void resizeItemsInternal(Integer nb_item);
  void computeMaterialIndexes(ComponentItemInternalData* item_internal_data, RunQueue& queue);
  void notifyLocalIdsChanged(Int32ConstArrayView old_to_new_ids);
  MeshComponentData* componentData() { return &m_data; }

  ConstArrayView<MeshMaterial*> trueMaterials()
  {
    return m_true_materials;
  }
  //@}

 private:

  //! Gestionnaire de matériaux
  IMeshMaterialMng* m_material_mng = nullptr;

  IUserMeshEnvironment* m_user_environment = nullptr;

  UniqueArray<IMeshMaterial*> m_materials;
  UniqueArray<MeshMaterial*> m_true_materials;

  //! Nombre total de mailles pour tous les matériaux
  Integer m_total_nb_cell_mat = 0;
  IItemGroupObserver* m_group_observer = nullptr;
  MeshComponentData m_data;
  MeshEnvironment* m_non_const_this = nullptr;
  InternalApi m_internal_api;

 public:

  void _computeMaterialIndexes(ComponentItemInternalData* item_internal_data, RunQueue& queue);
  void _computeMaterialIndexesMonoMat(ComponentItemInternalData* item_internal_data, RunQueue& queue);

 private:
  
  void _changeIds(MeshComponentData* component_data,Int32ConstArrayView old_to_new_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
