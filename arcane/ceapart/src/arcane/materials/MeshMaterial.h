﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterial.h                                              (C) 2000-2021 */
/*                                                                           */
/* Matériau d'un maillage.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIAL_H
#define ARCANE_MATERIALS_MESHMATERIAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ItemGroup.h"

#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MeshMaterialVariableIndexer.h"
#include "arcane/materials/MatItem.h"
#include "arcane/materials/MeshComponentData.h"

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
 public:

  MeshMaterial(MeshMaterialInfo* infos,MeshEnvironment* env,
               const String& name,Int32 mat_id);
  virtual ~MeshMaterial();

 public:

  IMeshMaterialMng* materialMng() override { return m_material_mng; }
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }
  MeshMaterialInfo* infos() const override { return m_infos; }
  const String& name() const override { return m_data.name(); }
  IMeshEnvironment* environment() const override;
  CellGroup cells() const override;

  MeshMaterialVariableIndexer* variableIndexer() override
  {
    return m_data.variableIndexer();
  }

  ConstArrayView<ComponentItemInternal*> itemsInternalView() const override
  {
    return m_data.itemsInternalView();
  }

  Int32 id() const override { return m_data.componentId(); }

  IUserMeshMaterial* userMaterial() const override { return m_user_material; }
  void setUserMaterial(IUserMeshMaterial* umm) override { m_user_material = umm; }

  MatCell findMatCell(AllEnvCell c) const override;
  ComponentCell findComponentCell(AllEnvCell c) const override;

  MatItemVectorView matView() override;
  ComponentItemVectorView view() override;

  void checkValid() override;

  bool isMaterial() const override { return true; }
  bool isEnvironment() const override { return false; }
  bool hasSpace(MatVarSpace space) const override { return space==MatVarSpace::MaterialAndEnvironment; }
  IMeshMaterial* asMaterial() override { return this; }
  IMeshEnvironment* asEnvironment() override { return nullptr; }

  ComponentPurePartItemVectorView pureItems() override;
  ComponentImpurePartItemVectorView impureItems() override;
  ComponentPartItemVectorView partItems(eMatPart part) override;

  MatPurePartItemVectorView pureMatItems() override;
  MatImpurePartItemVectorView impureMatItems() override;
  MatPartItemVectorView partMatItems(eMatPart part) override;

 public:

  ArrayView<ComponentItemInternal*> itemsInternalView()
  {
    return m_data.itemsInternalView();
  }

 public:

  //! Fonctions publiques mais réservées au IMeshMaterialMng
  //@{
  void build();
  void resizeItemsInternal(Integer nb_item);
  MeshComponentData* componentData() { return &m_data; }
  MeshEnvironment* trueEnvironment() { return m_environment; }
  //@}

 private:

  IMeshMaterialMng* m_material_mng;
  MeshMaterialInfo* m_infos;
  MeshEnvironment* m_environment;
  IUserMeshMaterial* m_user_material;
  MeshComponentData m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

