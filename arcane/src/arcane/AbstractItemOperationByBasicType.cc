// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractItemOperationByBasicType.cc                         (C) 2000-2016 */
/*                                                                           */
/* Opérateur abstrait sur des entités rangées par type.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/Item.h"
#include "arcane/AbstractItemOperationByBasicType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemOperationByBasicType::
applyVertex(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyLine2(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyTriangle3(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyQuad4(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyPentagon5(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyHexagon6(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO); 
}
void AbstractItemOperationByBasicType::
applyTetraedron4(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyPyramid5(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyPentaedron6(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyHexaedron8(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyHeptaedron10(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyOctaedron12(ItemVectorView items)
{
  ARCANE_UNUSED(items);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyHemiHexa7(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyHemiHexa6(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyHemiHexa5(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyAntiWedgeLeft6(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyAntiWedgeRight6(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyDiTetra5(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyDualNode(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyDualEdge(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO);
}
void AbstractItemOperationByBasicType::
applyDualFace(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO); 
}
void AbstractItemOperationByBasicType::
applyDualCell(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO); 
}
void AbstractItemOperationByBasicType::
applyLink(ItemVectorView group)
{
  ARCANE_UNUSED(group);
  throw NotImplementedException(A_FUNCINFO); 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

