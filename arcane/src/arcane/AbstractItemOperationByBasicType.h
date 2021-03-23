// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractItemOperationByBasicType.h                          (C) 2000-2012 */
/*                                                                           */
/* Opérateur abstrait sur des entités rangées par type.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ABSTRACTITEMOPERATIONBYBASICTYPE_H
#define ARCANE_ABSTRACTITEMOPERATIONBYBASICTYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IItemOperationByBasicType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IBase;
class IServiceInfo;
class IServiceFactory;
class ISubDomain;
class IApplication;
class ISession;
class ICaseOptions;

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

  //! Libère les ressources
  virtual ~AbstractItemOperationByBasicType() {}

 public:
  virtual void applyVertex(ItemVectorView items);
  virtual void applyLine2(ItemVectorView items);
  virtual void applyTriangle3(ItemVectorView items);
  virtual void applyQuad4(ItemVectorView items);
  virtual void applyPentagon5(ItemVectorView items);
  virtual void applyHexagon6(ItemVectorView items);
  virtual void applyTetraedron4(ItemVectorView items);
  virtual void applyPyramid5(ItemVectorView items);
  virtual void applyPentaedron6(ItemVectorView items);
  virtual void applyHexaedron8(ItemVectorView items);
  virtual void applyHeptaedron10(ItemVectorView items);
  virtual void applyOctaedron12(ItemVectorView items);
  virtual void applyHemiHexa7(ItemVectorView group);
  virtual void applyHemiHexa6(ItemVectorView group);
  virtual void applyHemiHexa5(ItemVectorView group);
  virtual void applyAntiWedgeLeft6(ItemVectorView group);
  virtual void applyAntiWedgeRight6(ItemVectorView group);
  virtual void applyDiTetra5(ItemVectorView group);
  virtual void applyDualNode(ItemVectorView group);
  virtual void applyDualEdge(ItemVectorView group);
  virtual void applyDualFace(ItemVectorView group);
  virtual void applyDualCell(ItemVectorView group);
  virtual void applyLink(ItemVectorView group);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

