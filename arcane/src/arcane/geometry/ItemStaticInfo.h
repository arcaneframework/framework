// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemStaticInfo.h                                            (C) 2000-2026 */
/*                                                                           */
/* Static information about entities.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_ITEMSTATICINFO_H
#define ARCANE_GEOMETRIC_ITEMSTATICINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Support for 7 and 8-faced prisms.
#ifndef __ARCANE_HAS_WEDGE_MESH_ELEMENT__
#define __ARCANE_HAS_WEDGE_MESH_ELEMENT__
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemStaticInfo
{
 public:

#ifdef __ARCANE_HAS_WEDGE_MESH_ELEMENT__

  static const Int32 __MAX_CELL2_EDGE = 6;


  //! Maximum number of nodes per cell
  static const Int32 __MAX_CELL_NODE = 12; // 8

  //! Maximum number of vertices per face
  static const Int32 __MAX_FACE_NODE = 6; // 4

  //! Maximum number of faces per cell
  static const Int32 __MAX_CELL_FACE = 8; // 6

  //! Maximum number of edges per cell
  static const Int32 __MAX_CELL_EDGE = 18; //  12
#else

  static const Int32 __MAX_CELL2_EDGE = 4;

  //! Maximum number of nodes per cell
  static const Int32 __MAX_CELL_NODE = 8;

  //! Maximum number of vertices per face
  static const Int32 __MAX_FACE_NODE = 4;

  //! Maximum number of faces per cell
  static const Int32 __MAX_CELL_FACE = 6;

  //! Maximum number of edges per cell
  static const Int32 __MAX_CELL_EDGE = 12;
#endif

  //! Number of element types
  static const Integer NB_TYPE = NB_BASIC_ITEM_TYPE;

  //! Maximum number of nodes per cell
  static const Integer MAX_CELL_NODE = __MAX_CELL_NODE;

  //! Maximum number of vertices per face
  static const Integer MAX_FACE_NODE = __MAX_FACE_NODE;

  //! Maximum number of faces per cell
  static const Integer MAX_CELL_FACE = __MAX_CELL_FACE;

  //! Maximum number of edges per cell
  static const Integer MAX_CELL_EDGE = __MAX_CELL_EDGE;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
