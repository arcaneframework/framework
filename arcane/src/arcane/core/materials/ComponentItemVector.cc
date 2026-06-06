// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVector.cc                                      (C) 2000-2024 */
/*                                                                           */
/* Vector over the entities of a component.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemVector.h"

#include "arcane/core/materials/IMeshComponent.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(IMeshComponent* component)
: m_p(component->_internalApi()->createItemVectorImpl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(ComponentItemVectorView rhs)
: m_p(rhs.component()->_internalApi()->createItemVectorImpl(rhs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setItems(SmallSpan<const Int32> local_ids)
{
  m_p->_setItems(local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView ComponentItemVector::
view() const
{
  return m_p->_view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView ComponentItemVector::
pureItems() const
{
  return m_p->_pureItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView ComponentItemVector::
impureItems() const
{
  return m_p->_impureItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemLocalIdListView ComponentItemVector::
_constituentItemListView() const
{
  return m_p->_constituentItemListView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<MatVarIndex> ComponentItemVector::
_matvarIndexes() const
{
  return m_p->_matvarIndexes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> ComponentItemVector::
_localIds() const
{
  return m_p->_localIds();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialMng* ComponentItemVector::
_materialMng() const
{
  return m_p->_materialMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshComponent* ComponentItemVector::
_component() const
{
  return m_p->_component();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
