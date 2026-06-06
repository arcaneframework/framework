// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedFace.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Semi-conforming mesh face.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_TIEDFACE_H
#define ARCANE_CORE_TIEDFACE_H
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
 * \brief Semi-conforming mesh face.
 */
class TiedFace
{
 public:

  TiedFace(Integer aindex, Face aface)
  : m_index(aindex)
  , m_face(aface)
  {
  }

  TiedFace() = default;

 public:

  //! Index of the face in the list of welded faces of the master face
  Integer index() const { return m_index; }

  //! Welded face
  Face face() const { return m_face; }

 private:

  //! Index of the face in the list of welded faces of the master face
  Integer m_index = NULL_ITEM_LOCAL_ID;
  //! Welded face
  Face m_face;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
