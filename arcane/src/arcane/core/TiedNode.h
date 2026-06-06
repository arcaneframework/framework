// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedNode.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Semi-conformal mesh node.                                                 */
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
 * \brief Semi-conformal mesh node.
 *
 * A semi-conformal mesh node is defined by the master face it belongs to
 * (see ITiedInterface) and not its iso-barycentric coordinates within that
 * face. These coordinates are always between -1 and 1, and their value depends
 * on the face type. For a 3D quadrilateral face, the definition is that of
 * GeometricUtilities::QuadMapping.
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

  //! Index of the node in the master face's tied nodes list
  Integer index() const { return m_index; }

  //! Tied node
  Node node() const { return m_node; }

  //! Iso-barycentric coordinates of the node
  Real2 isoCoordinates() const { return m_iso_coordinates; }

 private:

  //! Index of the node in the master face's tied nodes list
  Integer m_index = NULL_ITEM_LOCAL_ID;
  //! Tied node
  Node m_node;
  //! Iso-barycentric coordinates of the node
  Real2 m_iso_coordinates;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
