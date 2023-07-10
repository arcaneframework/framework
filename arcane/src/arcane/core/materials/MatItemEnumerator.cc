// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemEnumerator.cc                                        (C) 2000-2022 */
/*                                                                           */
/* Enumérateurs sur les mailles materiaux.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MatItemEnumerator.h"

#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshBlock.h"
#include "arcane/core/materials/MatItemVector.h"
#include "arcane/core/materials/EnvItemVector.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellEnumerator MatCellEnumerator::
create(IMeshMaterial* mat)
{
  return MatCellEnumerator(mat->_internalApi()->itemsInternalView(),
                           mat->variableIndexer()->matvarIndexes(),mat);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellEnumerator MatCellEnumerator::
create(const MatCellVector& miv)
{
  return create(miv.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellEnumerator MatCellEnumerator::
create(MatCellVectorView v)
{
  return MatCellEnumerator(v._itemsInternalView(),v._matvarIndexes(),v.component());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellEnumerator EnvCellEnumerator::
create(IMeshEnvironment* env)
{
  return EnvCellEnumerator(env->_internalApi()->itemsInternalView(),
                           env->variableIndexer()->matvarIndexes(),env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellEnumerator EnvCellEnumerator::
create(const EnvCellVector& miv)
{
  return create(miv.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellEnumerator EnvCellEnumerator::
create(EnvCellVectorView v)
{
  return EnvCellEnumerator(v._itemsInternalView(),v._matvarIndexes(),v.component());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellEnumerator CellGenericEnumerator::
create(IMeshEnvironment* env)
{
  return EnvCellEnumerator::create(env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellEnumerator CellGenericEnumerator::
create(const EnvCellVector& ecv)
{
  return EnvCellEnumerator::create(ecv);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellEnumerator CellGenericEnumerator::
create(EnvItemVectorView v)
{
  return EnvCellEnumerator::create(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellEnumerator CellGenericEnumerator::
create(IMeshMaterial* mat)
{
  return MatCellEnumerator::create(mat);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellEnumerator CellGenericEnumerator::
create(const MatCellVector& miv)
{
  return MatCellEnumerator::create(miv);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellEnumerator CellGenericEnumerator::
create(MatItemVectorView v)
{
  return MatCellEnumerator::create(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellEnumerator CellGenericEnumerator::
create(CellVectorView v)
{
  return CellEnumerator(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellEnumerator CellGenericEnumerator::
create(const CellGroup& v)
{
  return create(v.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentCellEnumerator ComponentCellEnumerator::
create(IMeshComponent* component)
{
  MeshMaterialVariableIndexer* vi = component->variableIndexer();

  return ComponentCellEnumerator(component->_internalApi()->itemsInternalView(),
                                 vi->matvarIndexes(),component);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentCellEnumerator ComponentCellEnumerator::
create(const ComponentItemVector& v)
{
  return create(v.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentCellEnumerator ComponentCellEnumerator::
create(ComponentItemVectorView v)
{
  return ComponentCellEnumerator(v._itemsInternalView(),v._matvarIndexes(),
                                 v.component());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentEnumerator::
ComponentEnumerator(ConstArrayView<IMeshComponent*> components)
: m_components(components)
, m_index(0)
, m_size(m_components.size())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatEnumerator::
MatEnumerator(IMeshMaterialMng* mm)
: m_mats(mm->materials())
, m_index(0)
, m_size(m_mats.size())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatEnumerator::
MatEnumerator(IMeshEnvironment* env)
: m_mats(env->materials())
, m_index(0)
, m_size(m_mats.size())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatEnumerator::
MatEnumerator(ConstArrayView<IMeshMaterial*> mats)
: m_mats(mats)
, m_index(0)
, m_size(m_mats.size())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvEnumerator::
EnvEnumerator(IMeshMaterialMng* mm)
: m_envs(mm->environments())
, m_index(0)
, m_size(m_envs.size())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvEnumerator::
EnvEnumerator(IMeshBlock* mb)
: m_envs(mb->environments())
, m_index(0)
, m_size(m_envs.size())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvEnumerator::
EnvEnumerator(ConstArrayView<IMeshEnvironment*> envs)
: m_envs(envs)
, m_index(0)
, m_size(m_envs.size())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvCellEnumerator AllEnvCellEnumerator::
create(AllEnvCellVectorView items)
{
  return AllEnvCellEnumerator(items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvCellEnumerator AllEnvCellEnumerator::
create(IMeshMaterialMng* mng,const CellVectorView& view)
{
  return create(mng->view(view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvCellEnumerator AllEnvCellEnumerator::
create(IMeshMaterialMng* mng,const CellGroup& group)
{
  return create(mng->view(group));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvCellEnumerator AllEnvCellEnumerator::
create(IMeshBlock* block)
{
  return create(block->view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPartCellEnumerator ComponentPartCellEnumerator::
create(ComponentPartItemVectorView v)
{
  return ComponentPartCellEnumerator(v.component(),v.componentPartIndex(),
                                     v.valueIndexes(),v.itemIndexes(),
                                     v.itemsInternal(),0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPartCellEnumerator ComponentPartCellEnumerator::
create(IMeshComponent* component,eMatPart part)
{
  return create(component->partItems(part));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPartCellEnumerator::
ComponentPartCellEnumerator(const ComponentPartItemVectorView& v,Integer base_index)
: m_index(0)
, m_size(v.itemIndexes().size())
, m_var_idx(v.componentPartIndex())
, m_base_index(base_index)
, m_value_indexes(v.valueIndexes())
, m_item_indexes(v.itemIndexes())
, m_items_internal(v.itemsInternal())
, m_component(v.component())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatPartCellEnumerator::
MatPartCellEnumerator(const MatPartItemVectorView& v)
: ComponentPartCellEnumerator(v,0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatPartCellEnumerator MatPartCellEnumerator::
create(IMeshMaterial* mat,eMatPart part)
{
  MatPartItemVectorView v(mat->partMatItems(part));
  return MatPartCellEnumerator(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatPartCellEnumerator MatPartCellEnumerator::
create(MatPartItemVectorView v)
{
  return MatPartCellEnumerator(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvPartCellEnumerator::
EnvPartCellEnumerator(const EnvPartItemVectorView& v)
: ComponentPartCellEnumerator(v,0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvPartCellEnumerator EnvPartCellEnumerator::
create(IMeshEnvironment* env,eMatPart part)
{
  EnvPartItemVectorView v(env->partEnvItems(part));
  return EnvPartCellEnumerator(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvPartCellEnumerator EnvPartCellEnumerator::
create(EnvPartItemVectorView v)
{
  return EnvPartCellEnumerator(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
