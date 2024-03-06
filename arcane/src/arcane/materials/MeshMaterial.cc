// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterial.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Matériau d'un maillage.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentItemVectorView.h"
#include "arcane/materials/MeshComponentPartData.h"
#include "arcane/materials/ComponentPartItemVectorView.h"

#include "arcane/materials/internal/MeshMaterial.h"
#include "arcane/materials/internal/MeshEnvironment.h"
#include "arcane/materials/internal/ConstituentItemVectorImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterial::
MeshMaterial(MeshMaterialInfo* infos, MeshEnvironment* env,
             const String& name, Int16 mat_id)
: TraceAccessor(infos->materialMng()->traceMng())
, m_material_mng(infos->materialMng())
, m_infos(infos)
, m_environment(env)
, m_user_material(nullptr)
, m_data(this, name, mat_id, m_material_mng->_internalApi()->componentItemSharedInfo(LEVEL_MATERIAL), true)
, m_non_const_this(this)
, m_internal_api(this)
{
  if (!env)
    ARCANE_THROW(ArgumentException, "null environement for material '{0}'", name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterial::
build()
{
  IMesh* mesh = m_material_mng->mesh();
  IItemFamily* cell_family = mesh->cellFamily();
  String group_name = m_material_mng->name() + "_" + name();
  CellGroup items = cell_family->findGroup(group_name, true);
  m_data._setItems(items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshEnvironment* MeshMaterial::
environment() const
{
  return m_environment;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCell MeshMaterial::
findMatCell(AllEnvCell c) const
{
  Int32 mat_id = m_data.componentId();
  ENUMERATE_CELL_ENVCELL (ienvcell, c) {
    ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
      MatCell mc = *imatcell;
      Int32 mid = mc.materialId();
      if (mid == mat_id)
        return mc;
    }
  }
  return MatCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentCell MeshMaterial::
findComponentCell(AllEnvCell c) const
{
  return findMatCell(c);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatItemVectorView MeshMaterial::
matView() const
{
  return { m_non_const_this, variableIndexer()->matvarIndexes(),
           constituentItemListView(), variableIndexer()->localIds() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView MeshMaterial::
view() const
{
  return matView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterial::
resizeItemsInternal(Integer nb_item)
{
  m_data._resizeItemsInternal(nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterial::
checkValid()
{
  m_data.checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup MeshMaterial::
cells() const
{
  return m_data.items();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView MeshMaterial::
pureItems() const
{
  return m_data._partData()->pureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView MeshMaterial::
impureItems() const
{
  return m_data._partData()->impureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPartItemVectorView MeshMaterial::
partItems(eMatPart part) const
{
  return m_data._partData()->partView(part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatPurePartItemVectorView MeshMaterial::
pureMatItems() const
{
  return { m_non_const_this, m_data._partData()->pureView() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatImpurePartItemVectorView MeshMaterial::
impureMatItems() const
{
  return { m_non_const_this, m_data._partData()->impureView() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatPartItemVectorView MeshMaterial::
partMatItems(eMatPart part) const
{
  return { m_non_const_this, m_data._partData()->partView(part) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MeshMaterial::InternalApi::
variableIndexerIndex() const
{
  return variableIndexer()->index();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IConstituentItemVectorImpl> MeshMaterial::InternalApi::
createItemVectorImpl() const
{
  auto* x = new ConstituentItemVectorImpl(m_material->m_non_const_this);
  return makeRef<IConstituentItemVectorImpl>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IConstituentItemVectorImpl> MeshMaterial::InternalApi::
createItemVectorImpl(ComponentItemVectorView rhs) const
{
  auto* x = new ConstituentItemVectorImpl(rhs);
  return makeRef<IConstituentItemVectorImpl>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
