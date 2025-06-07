// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractItemOperationByBasicType.h                          (C) 2000-2025 */
/*                                                                           */
/* Opérateur abstrait sur des entités rangées par type.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ABSTRACTITEMOPERATIONBYBASICTYPE_H
#define ARCANE_CORE_ABSTRACTITEMOPERATIONBYBASICTYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IItemOperationByBasicType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Opérateur abstrait sur des entités rangées par type.
 */
class ARCANE_CORE_EXPORT AbstractItemOperationByBasicType
: public IItemOperationByBasicType
{
 public:

  void applyVertex(ItemVectorView items) override;
  void applyLine2(ItemVectorView items) override;
  void applyTriangle3(ItemVectorView items) override;
  void applyQuad4(ItemVectorView items) override;
  void applyPentagon5(ItemVectorView items) override;
  void applyHexagon6(ItemVectorView items) override;
  void applyTetraedron4(ItemVectorView items) override;
  void applyPyramid5(ItemVectorView items) override;
  void applyPentaedron6(ItemVectorView items) override;
  void applyHexaedron8(ItemVectorView items) override;
  void applyHeptaedron10(ItemVectorView items) override;
  void applyOctaedron12(ItemVectorView items) override;
  void applyHemiHexa7(ItemVectorView group) override;
  void applyHemiHexa6(ItemVectorView group) override;
  void applyHemiHexa5(ItemVectorView group) override;
  void applyAntiWedgeLeft6(ItemVectorView group) override;
  void applyAntiWedgeRight6(ItemVectorView group) override;
  void applyDiTetra5(ItemVectorView group) override;
  void applyDualNode(ItemVectorView group) override;
  void applyDualEdge(ItemVectorView group) override;
  void applyDualFace(ItemVectorView group) override;
  void applyDualCell(ItemVectorView group) override;
  void applyLink(ItemVectorView group) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

