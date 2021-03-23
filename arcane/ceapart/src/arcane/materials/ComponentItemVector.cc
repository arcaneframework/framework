// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVector.cc                                      (C) 2000-2017 */
/*                                                                           */
/* Vecteur sur les entités d'un composant.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"

#include "arcane/materials/ComponentItemVector.h"
#include "arcane/materials/IMeshComponent.h"
#include "arcane/materials/MeshComponentPartData.h"

#include "arcane/utils/FatalErrorException.h"

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

ComponentItemVector::Impl::
Impl(IMeshComponent* component)
: m_material_mng(component->materialMng())
, m_component(component)
, m_part_data(new MeshComponentPartData(component))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::Impl::
Impl(Impl&& rhs)
: m_material_mng(rhs.m_material_mng)
, m_component(rhs.m_component)
, m_items_internal(rhs.m_items_internal)
, m_matvar_indexes(rhs.m_matvar_indexes)
, m_part_data(new MeshComponentPartData(std::move(*rhs.m_part_data)))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::Impl::
Impl(IMeshComponent* component,ConstArrayView<ComponentItemInternal*> items_internal,
     ConstArrayView<MatVarIndex> matvar_indexes)
: m_material_mng(component->materialMng())
, m_component(component)
, m_items_internal(items_internal)
, m_matvar_indexes(matvar_indexes)
, m_part_data(new MeshComponentPartData(component))
{
  m_part_data->setFromMatVarIndexes(matvar_indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::Impl::
~Impl()
{
  delete m_part_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::Impl::
deleteMe()
{
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(IMeshComponent* component)
: m_p(new Impl(component))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(ComponentItemVectorView rhs)
: m_p(new Impl(rhs.component(),rhs.itemsInternalView(),rhs.matvarIndexes()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setItemsInternal(ConstArrayView<ComponentItemInternal*> globals,
                  ConstArrayView<ComponentItemInternal*> multiples)
{
  Integer nb_global = globals.size();
  Integer nb_multiple = multiples.size();

  m_p->m_items_internal.resize(nb_global+nb_multiple);

  m_p->m_items_internal.subView(0,nb_global).copy(globals);
  m_p->m_items_internal.subView(nb_global,nb_multiple).copy(multiples);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                  ConstArrayView<MatVarIndex> multiples)
{
  Integer nb_global = globals.size();
  Integer nb_multiple = multiples.size();

  m_p->m_matvar_indexes.resize(nb_global+nb_multiple);

  m_p->m_matvar_indexes.subView(0,nb_global).copy(globals);
  m_p->m_matvar_indexes.subView(nb_global,nb_multiple).copy(multiples);

  {
    Int32Array& idx = m_p->m_part_data->_mutableValueIndexes(eMatPart::Pure);
    idx.resize(nb_global);
    for( Integer i=0; i<nb_global; ++i )
      idx[i] = globals[i].valueIndex();
  }

  {
    Int32Array& idx = m_p->m_part_data->_mutableValueIndexes(eMatPart::Impure);
    idx.resize(nb_multiple);
    for( Integer i=0; i<nb_multiple; ++i )
      idx[i] = multiples[i].valueIndex();
  }

  m_p->m_part_data->_notifyValueIndexesChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView ComponentItemVector::
pureItems() const
{
  return m_p->m_part_data->pureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView ComponentItemVector::
impureItems() const
{
  return m_p->m_part_data->impureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
