// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironment.h                                           (C) 2000-2023 */
/*                                                                           */
/* Milieu d'un maillage.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHENVIRONMENT_H
#define ARCANE_MATERIALS_INTERNAL_MESHENVIRONMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ItemGroup.h"
#include "arcane/MeshVariableScalarRef.h"
#include "arcane/VariableTypedef.h"

#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/MatItemInternal.h"
#include "arcane/materials/MeshComponentData.h"

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
 public:

  MeshEnvironment(IMeshMaterialMng* mm,const String& name,Int32 env_id);
  virtual ~MeshEnvironment();

 public:

  IMeshMaterialMng* materialMng() override { return m_material_mng; }
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }
  const String& name() const override { return m_data.name(); }
  CellGroup cells() const override { return m_data.items(); }
  ConstArrayView<IMeshMaterial*> materials() override
  {
    return m_materials;
  }
  Integer nbMaterial() const override
  {
    return m_materials.size();
  }
  MeshMaterialVariableIndexer* variableIndexer() const override
  {
    return m_data.variableIndexer();
  }
  ConstArrayView<ComponentItemInternal*> itemsInternalView() const override
  {
    return m_data.itemsInternalView();
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

  ArrayView<ComponentItemInternal*> itemsInternalView()
  {
    return m_data.itemsInternalView();
  }

 public:

  //! Fonctions publiques mais réservées au IMeshMaterialMng
  //@{
  void build();
  void addMaterial(MeshMaterial* mm);
  void setVariableIndexer(MeshMaterialVariableIndexer* idx);
  //! Recalcule le nombre de mailles par matériau et de mailles totales
  void computeNbMatPerCell();
  
  void computeItemListForMaterials(const VariableCellInt32& nb_env_per_cell);

  //! Nombre total de mailles pour tous les matériaux
  Integer totalNbCellMat() const { return m_total_nb_cell_mat; }

  void resizeItemsInternal(Integer nb_item);
  void computeMaterialIndexes(ComponentItemInternalData* item_internal_data);
  void notifyLocalIdsChanged(Int32ConstArrayView old_to_new_ids);
  MeshComponentData* componentData() { return &m_data; }

  void updateItemsDirect(const VariableCellInt32& nb_env_per_cell,MeshMaterial* mat,
                         Int32ConstArrayView local_ids,eOperation operation,bool add_to_env_indexer=false);

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

 public:

  //! Nombre de matériaux de ce milieu par maille
  VariableCellInt32 m_nb_mat_per_cell;

 private:
  
  //! Nombre total de mailles pour tous les matériaux
  Integer m_total_nb_cell_mat = 0;
  IItemGroupObserver* m_group_observer = nullptr;
  MeshComponentData m_data;
  MeshEnvironment* m_non_const_this = nullptr;

 private:
  
  void _changeIds(MeshComponentData* component_data,Int32ConstArrayView old_to_new_ids);
  void _addItemsToIndexer(const VariableCellInt32& nb_env_per_cell,
                          MeshMaterialVariableIndexer* var_indexer,
                          Int32ConstArrayView local_ids);
  void _removeItemsDirect(MeshMaterial* mat,Int32ConstArrayView local_ids,
                          bool update_env_indexer);
  void _addItemsDirect(const VariableCellInt32& nb_env_per_cell,MeshMaterial* mat,
                       Int32ConstArrayView local_ids,bool update_env_indexer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

