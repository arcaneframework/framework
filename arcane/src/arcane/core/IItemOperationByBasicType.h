// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemOperationByBasicType.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface d'un opérateur sur des entités rangées par type.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMOPERATIONBYBASICTYPE_H
#define ARCANE_CORE_IITEMOPERATIONBYBASICTYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ItemVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Interface d'un opérateur sur des entités rangées par type.
 */
class ARCANE_CORE_EXPORT IItemOperationByBasicType
{
 public:

  //! Libère les ressources
  virtual ~IItemOperationByBasicType() = default;

 public:

  virtual void applyVertex(ItemVectorView items) = 0;
  virtual void applyLine2(ItemVectorView items) = 0;
  virtual void applyTriangle3(ItemVectorView items) = 0;
  virtual void applyQuad4(ItemVectorView items) = 0;
  virtual void applyPentagon5(ItemVectorView items) = 0;
  virtual void applyHexagon6(ItemVectorView items) = 0;
  virtual void applyTetraedron4(ItemVectorView items) = 0;
  virtual void applyPyramid5(ItemVectorView items) = 0;
  virtual void applyPentaedron6(ItemVectorView items) = 0;
  virtual void applyHexaedron8(ItemVectorView items) = 0;
  virtual void applyHeptaedron10(ItemVectorView items) = 0;
  virtual void applyOctaedron12(ItemVectorView items) = 0;
  virtual void applyHemiHexa7(ItemVectorView group) = 0;
  virtual void applyHemiHexa6(ItemVectorView group) = 0;
  virtual void applyHemiHexa5(ItemVectorView group) = 0;
  virtual void applyAntiWedgeLeft6(ItemVectorView group) = 0;
  virtual void applyAntiWedgeRight6(ItemVectorView group) = 0;
  virtual void applyDiTetra5(ItemVectorView group) = 0;
  virtual void applyDualNode(ItemVectorView group) = 0;
  virtual void applyDualEdge(ItemVectorView group) = 0;
  virtual void applyDualFace(ItemVectorView group) = 0;
  virtual void applyDualCell(ItemVectorView group) = 0;
  virtual void applyLink(ItemVectorView group) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

