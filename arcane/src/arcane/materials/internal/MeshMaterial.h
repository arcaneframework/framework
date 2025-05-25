// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterial.h                                              (C) 2000-2025 */
/*                                                                           */
/* Matériau d'un maillage.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIAL_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

#include "arcane/materials/internal/MeshComponentData.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class MeshEnvironment;
class MatItemVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Matériau d'un maillage.
 *
 * Les matériaux sont créés via IMeshMaterialMng::createMaterial().
 * Les matériaux ne peuvent pas être détruits et tous les matériaux doivent
 * être créés lors de l'initialisation. Un matériau peut n'avoir aucune
 * maille.
 */
class MeshMaterial
: public TraceAccessor
, public IMeshMaterial
{
  class InternalApi
  : public IMeshComponentInternal
  {
   public:

    explicit InternalApi(MeshMaterial* mat)
    : m_material(mat)
    {}

   public:

    MeshMaterialVariableIndexer* variableIndexer() const override
    {
      return m_material->variableIndexer();
    }
    ConstituentItemLocalIdListView constituentItemListView() const override
    {
      return m_material->constituentItemListView();
    }
    Int32 variableIndexerIndex() const override;
    Ref<IConstituentItemVectorImpl> createItemVectorImpl() const override;
    Ref<IConstituentItemVectorImpl> createItemVectorImpl(ComponentItemVectorView rhs) const override;

   private:

    MeshMaterial* m_material = nullptr;
  };

 public:

  MeshMaterial(MeshMaterialInfo* infos,MeshEnvironment* env,
               const String& name,Int16 mat_id);

 public:

  IMeshMaterialMng* materialMng() override { return m_material_mng; }
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }
  MeshMaterialInfo* infos() const override { return m_infos; }
  String name() const override { return m_data.name(); }
  IMeshEnvironment* environment() const override;
  CellGroup cells() const override;

  MeshMaterialVariableIndexer* variableIndexer() const
  {
    return m_data.variableIndexer();
  }

  ConstituentItemLocalIdListView constituentItemListView() const
  {
    return m_data.constituentItemListView();
  }

  Int32 id() const override { return m_data.componentId(); }

  IUserMeshMaterial* userMaterial() const override { return m_user_material; }
  void setUserMaterial(IUserMeshMaterial* umm) override { m_user_material = umm; }

  MatCell findMatCell(AllEnvCell c) const override;
  ComponentCell findComponentCell(AllEnvCell c) const override;

  MatItemVectorView matView() const override;
  ComponentItemVectorView view() const override;

  void checkValid() override;

  bool isMaterial() const override { return true; }
  bool isEnvironment() const override { return false; }
  bool hasSpace(MatVarSpace space) const override { return space==MatVarSpace::MaterialAndEnvironment; }
  IMeshMaterial* asMaterial() override { return this; }
  IMeshEnvironment* asEnvironment() override { return nullptr; }

  ComponentPurePartItemVectorView pureItems() const override;
  ComponentImpurePartItemVectorView impureItems() const override;
  ComponentPartItemVectorView partItems(eMatPart part) const override;

  MatPurePartItemVectorView pureMatItems() const override;
  MatImpurePartItemVectorView impureMatItems() const override;
  MatPartItemVectorView partMatItems(eMatPart part) const override;

  void setSpecificExecutionPolicy(Accelerator::eExecutionPolicy policy) override
  {
    m_data.setSpecificExecutionPolicy(policy);
  }
  Accelerator::eExecutionPolicy specificExecutionPolicy() const override
  {
    return m_data.specificExecutionPolicy();
  }

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
  void resizeItemsInternal(Integer nb_item);
  MeshComponentData* componentData() { return &m_data; }
  MeshEnvironment* trueEnvironment() { return m_environment; }
  const MeshEnvironment* trueEnvironment() const { return m_environment; }
  //@}

 private:

  IMeshMaterialMng* m_material_mng = nullptr;
  MeshMaterialInfo* m_infos = nullptr;
  MeshEnvironment* m_environment = nullptr;
  IUserMeshMaterial* m_user_material = nullptr;
  MeshComponentData m_data;
  MeshMaterial* m_non_const_this = nullptr;
  InternalApi m_internal_api;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

