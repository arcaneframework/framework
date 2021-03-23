// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeView.h                                             (C) 2000-2016 */
/*                                                                           */
/* Gestion des formes géométriques 2D et 3D.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMSHAPEVIEW_H
#define ARCANE_GEOMETRIC_GEOMSHAPEVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Real3.h"

#include "arcane/Item.h"

#include "arcane/geometric/GeometricConnectic.h"
#include "arcane/geometric/GeomElement.h"
#include "arcane/geometric/CellConnectivity.h"
#include "arcane/geometric/GeomShapeMutableView.h"

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
 * \brief Vue constante sur une forme géométrique GeomShape.
 *
 * Une vue sur une forme géométrique permet de récupérer de manière
 * optimisée la position des noeuds, des faces et des arêtes (en 3D)
 * d'un objet géométrique.
 *
 * Cette classe gère une vue sur une forme géométrique. Il existe deux
 * manières d'initialiser une vue:
 * - en récupérant la vue associée à une maille du maillage via l'appel
 * à GeomShapeMng::initShape(). Dans ce cas, il est possible de
 * récupérer la maille associée via la méthode cell().
 * - à partir d'une instance temporaire de GeomShape
 * via une des deux méthodes initFromHexa() ou initFromQuad().
 *
 */
class ARCANE_CEA_GEOMETRIC_EXPORT GeomShapeView
{
  friend class GeomShapeMng;
  friend class GeomShape;
  friend class GeomShapeConnectivity;

 private:
  
  static CellConnectivity* global_cell_connectivity[NB_BASIC_ITEM_TYPE];
  static GeomShapeConnectivity* global_connectivity;

 public:

  static void initializeConnectivity();

 public:
  
  GeomShapeView()
  : m_node_ptr(0), m_face_ptr(0), m_center_ptr(0),
    m_cell_connectivity(global_cell_connectivity[IT_NullType]),
    m_item_internal(ItemInternal::nullItem())
  {
  }

 public:

#include "arcane/geometric/GeneratedGeomShapeViewDeclarations.h"

 public:

  //! Remplit \a hexa avec les informations du \a i-ème sous-volume de contrôle
  void fillSubZoneElement(HexaElementView hexa, Integer i);
  //! Remplit \a quad avec les informations du \a i-ème sous-volume de contrôle
  void fillSubZoneElement(QuadElementView quad, Integer i);

  /*!
   * \deprecated Utiliser GeomShape::initFromHexaedron8() à la place.
   */
  ARCANE_DEPRECATED_122 void initFromHexa(HexaElementConstView hexa,GeomShape& geom_cell);
  /*!
   * \deprecated Utiliser GeomShape::initFromQuad4() à la place.
   */
  ARCANE_DEPRECATED_122 void initFromQuad(QuadElementConstView hexa,GeomShape& geom_cell);

 public:

  /*!
   * \name Récupération des coordonnées.
   */
  //@{
  //! Position du \a ième noeud de la forme
  const Real3 node(Integer i) const
  {
    return m_node_ptr[i];
  }
  
  //! Position du centre de la \a ième face de la forme
  const Real3 face(Integer i) const
  {
    return m_face_ptr[i];
  }

  //! Position du centre de la forme
  const Real3 center() const
  {
    return *m_center_ptr;
  }

  //! Position du centre de la \a i-ème arête de la forme
  inline const Real3 edge(Integer i) const
  {
    return 0.5 * (node(m_cell_connectivity->m_edge_direct_connectic[(i*2)]) + node(m_cell_connectivity->m_edge_direct_connectic[(i*2)+1]));
  }
  //@}

  //! Entité associée (null si aucune)
  Item item() const { return Item(m_item_internal); }
  //! Maille associée (null si aucune)
  Cell cell() const { return Cell(m_item_internal); }
  //! Face associée (null si aucune)
  Face face() const { return Face(m_item_internal); }

 protected:

  void _setArray(const Real3* node_ptr,const Real3* face_ptr,const Real3* center_ptr)
  {
    m_node_ptr = node_ptr;
    m_face_ptr = face_ptr;
    m_center_ptr = center_ptr;
  }

 private:

  ARCANE_RESTRICT const Real3* m_node_ptr;
  ARCANE_RESTRICT const Real3* m_face_ptr;
  ARCANE_RESTRICT const Real3* m_center_ptr;
  //! Informations sur la connectivité
  CellConnectivity* m_cell_connectivity;
  //! Information sur l'entité d'origine (ItemInternal::nullItem() si aucune)
  ItemInternal* m_item_internal;

 protected:

  //TODO: A SUPPRIMER
  const Real3POD* _nodeView() const { return (Real3POD*)m_node_ptr; }

 public:


 public:
  /*!
   * \name Informations sur la connectivité.
   */
  //@{
  //! Informations de connectivité aux noeuds.
  const NodeConnectic& nodeConnectic(Integer i) const
  {
    return m_cell_connectivity->nodeConnectic()[i];
  }

  //! Informations de connectivité aux arêtes.
  const EdgeConnectic& edgeConnectic(Integer i) const
  {
    return m_cell_connectivity->edgeConnectic()[i];
  }

  //! Informations de connectivité aux faces
  const FaceConnectic& faceConnectic(Integer i) const
  {
    return m_cell_connectivity->faceConnectic()[i];
  }

  //! Nombre de sous volume de controle
  Integer nbSubZone() const
  {
    return m_cell_connectivity->nbSubZone();
  }

  //! Nombre de sous faces internes SVC
  Integer nbSvcFace() const
  {
    return m_cell_connectivity->nbSubZoneFace();
  }

  //! Numéro local du sommet associé au sous volume de controle
  Integer nodeAssociation(Integer i) const
  {
    return m_cell_connectivity->nodeAssociation()[i];
  }

  const SVCFaceConnectic& svcFaceConnectic(Integer i) const
  {
    return m_cell_connectivity->SCVFaceConnectic()[i];
  }

  //! Numéro locaux dans le sous-volumes de contrôle
  Integer edgeNodeSubZoneId(Integer i) const
  {
    return m_cell_connectivity->m_edge_node_sub_zone_id[i];
  }

  Integer faceNodeSubZoneId(Integer i) const
  {
    return m_cell_connectivity->m_face_node_sub_zone_id[i];
  }
  //@}

  /*!
   * \brief Type géométrique de la forme.
   *
   * Si la forme est assossiée à une entité (récupérable via item()),
   * il s'agit aussi du type de l'entité.
   *
   * Retourne \a GeomType::NullType si l'instance n'est pas initialisée.
   */
  GeomType geomType() const
  {
    return m_cell_connectivity->cellType();
  }

 protected:

  void _setItem(Item item)
  {
    m_cell_connectivity = global_cell_connectivity[item.type()];
    m_item_internal = item.internal();
  }

  void _setNullItem(int item_type)
  {
    m_item_internal = ItemInternal::nullItem();
    m_cell_connectivity = global_cell_connectivity[item_type];
  }

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//TODO: Utiliser des Traits pour le nombre de noeuds et le nom du SVC.
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Vue sur forme géométrique 2D.
 */
class GeomShape2DView
: public GeomShapeView
{
 public:
  GeomShape2DView(){}
  explicit GeomShape2DView(const GeomShapeView& rhs) : GeomShapeView(rhs){}
};

/*!
 * \ingroup ArcaneGeometric
 * \brief Vue sur forme géométrique 3D.
 */
class GeomShape3DView
: public GeomShapeView
{
 public:
  GeomShape3DView(){}
  explicit GeomShape3DView(const GeomShapeView& rhs) : GeomShapeView(rhs){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometric/GeneratedGeomShapeView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! A supprimer à terme
#include "arcane/geometric/GeomShape.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
