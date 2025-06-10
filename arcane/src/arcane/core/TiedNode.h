// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedNode.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Noeud semi-conforme du maillage.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_TIEDNODE_H
#define ARCANE_CORE_TIEDNODE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Noeud semi-conforme du maillage.
 *
 * Un noeud semi-conforme du maillage est défini par la face maître
 * auquel il appartient (voir ITiedInterface) et pas ses coordonnées
 * iso-barycentriques dans cette face. Ces coordonnées sont toujours comprises
 * entre -1 et 1 et leur valeur dépend du type de la face. Pour une face
 * 3D quadrangulaire, la définition est celle de GeometricUtilities::QuadMapping.
 */
class TiedNode
{
 public:
 public:

  TiedNode(Integer aindex, Node anode, Real2 iso_coordinates)
  : m_index(aindex)
  , m_node(anode)
  , m_iso_coordinates(iso_coordinates)
  {
  }

  TiedNode() = default;

 public:

  //! Indice du noeud dans la liste des noeuds soudés de la face maitre
  Integer index() const { return m_index; }

  //! Noeud lié
  Node node() const { return m_node; }

  //! Coordonnées iso-barycentriques du noeud
  Real2 isoCoordinates() const { return m_iso_coordinates; }

 private:

  //! Indice du noeud dans la liste des noeuds soudés de la face maitre
  Integer m_index = NULL_ITEM_LOCAL_ID;
  //! Noeud lié
  Node m_node;
  //! Coordonnées iso-barycentriques du noeud
  Real2 m_iso_coordinates;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

