// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShape.h                                                 (C) 2000-2015 */
/*                                                                           */
/* Forme géométrique.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMSHAPE_H
#define ARCANE_GEOMETRIC_GEOMSHAPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometric/GeomShapeView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Forme géométrique.
 *
 * Cette classe ne s'utilise que pour créer une forme géométrique temporaire.
 * Pour une forme géométrique issue d'une maille, il faut passer par le
 * GeomShapeMng.
 *
 * Une instance de cette classe permet de conserver les informations
 * nécessaires pour une forme géométrique.
 *
 * Il est possible d'initialiser directement
 * une forme géométrique à partir d'un hexaèdre (initFromHexaedron8())
 * ou d'un quadrangle (initFromQuad4()). Ces méthodes initialisent la forme
 * et retournent une vue dessus.
 * \code
 * GeomShape shape;
 * HexaElement hexa;
 * GeomShapeView shape_view = shape.initFromHexaedron8(hexa);
 * \endcode
 *
 * \todo mettre en place une initialisation spécifique. Pour cela, il faudra
 * utiliser le toMutableView() mais il faut aussi pouvoir spécifier le geomType().
 */
class ARCANE_CEA_GEOMETRIC_EXPORT GeomShape
{
  // TEMPORAIRE: a supprimer quand les initFromHexa() et initFromQuad()
  // de GeomShapeView seront supprimés
  friend class GeomShapeView;

 public:
  
  //! Vue modifiable sur cet instance.
  GeomShapeMutableView toMutableView()
  {
    return GeomShapeMutableView((Real3*)m_node_ptr,(Real3*)m_face_ptr,(Real3*)&m_center);
  }

  //! Initialise la forme avec un hexaèdre \a hexa et retourne une vue dessus.
  Hexaedron8ShapeView initFromHexaedron8(Hexaedron8ElementConstView hexa);

  //! Initialise la forme avec un quadrangle \a quad et retourne une vue dessus.
  Quad4ShapeView initFromQuad4(Quad4ElementConstView quad);

 protected:

  void _setArray(GeomShapeView& shape)
  {
    shape._setArray((Real3*)m_node_ptr,(Real3*)m_face_ptr,(Real3*)&m_center);
  }

 private:

  Real3POD m_node_ptr[ItemStaticInfo::MAX_CELL_NODE];
  Real3POD m_face_ptr[ItemStaticInfo::MAX_CELL_FACE];
  Real3POD m_center;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
