// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeMutableView.h                                      (C) 2000-2014 */
/*                                                                           */
/* Vue modifiable sur un GeomShape.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMSHAPEMUTABLEVIEW_H
#define ARCANE_GEOMETRIC_GEOMSHAPEMUTABLEVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Real3.h"

#include "arcane/Item.h"

#include "arcane/geometric/GeometricConnectic.h"
#include "arcane/geometric/GeomElement.h"
#include "arcane/geometric/CellConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GeomShapeConnectivity;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Vue modifiable sur un GeomShape.
 *
 * Une instance de cette classe permet de modifier les coordonnées
 * des noeuds, des faces et du centre d'un GeomShape.
 *
 * Pour récupérer une instance de cet objet, il faut appeler
 * GeomShapeMng::mutableShapeView() ou GeomShape::toMutableView().
 */
class GeomShapeMutableView
{
 public:

  friend class GeomShapeMng;
  friend class GeomShape;

 public:

  GeomShapeMutableView()
  : m_node_ptr(0), m_face_ptr(0), m_center_ptr(0){}

 private:

  GeomShapeMutableView(Real3* node_ptr,Real3* face_ptr,Real3* center_ptr)
  : m_node_ptr(node_ptr), m_face_ptr(face_ptr), m_center_ptr(center_ptr){}

 public:
  
  inline const Real3 node(Integer id) const
  {
    return m_node_ptr[id];
  }

  inline const Real3 face(Integer id) const
  {
    return m_face_ptr[id];
  }

  inline const Real3 center() const
  {
    return *m_center_ptr;
  }

  inline void setNode(Integer id,const Real3& v)
  {
    m_node_ptr[id] = v;
  }

  inline void setFace(Integer id,const Real3& v)
  {
    m_face_ptr[id] = v;
  }

  inline void setCenter(const Real3& v)
  {
    (*m_center_ptr) = v;
  }

 private:

  Real3* m_node_ptr;
  Real3* m_face_ptr;
  Real3* m_center_ptr;

 private:

  inline void _setArray(Real3* node_ptr,Real3* face_ptr,Real3* center_ptr)
  {
    m_node_ptr = node_ptr;
    m_face_ptr = face_ptr;
    m_center_ptr = center_ptr;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
