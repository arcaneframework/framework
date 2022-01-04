// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemStaticInfo.h                                            (C) 2000-2014 */
/*                                                                           */
/* Informations statiques sur les entités.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_ITEMSTATICINFO_H
#define ARCANE_GEOMETRIC_ITEMSTATICINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Prise en compte des prismes a 7 et 8 faces.
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


  //! Nombre maximum de noeuds par maille
  static const Int32 __MAX_CELL_NODE = 12; // 8

  //! Nombre maximum de sommets par face
  static const Int32 __MAX_FACE_NODE = 6; // 4

  //! Nombre maximum de faces par maille
  static const Int32 __MAX_CELL_FACE = 8; // 6

  //! Nombre maximum d'arêtes par maille
  static const Int32 __MAX_CELL_EDGE = 18; //  12
#else

  static const Int32 __MAX_CELL2_EDGE = 4;

  //! Nombre maximum de noeuds par maille
  static const Int32 __MAX_CELL_NODE = 8;

  //! Nombre maximum de sommets par face
  static const Int32 __MAX_FACE_NODE = 4;

  //! Nombre maximum de faces par maille
  static const Int32 __MAX_CELL_FACE = 6;

  //! Nombre maximum d'arêtes par maille
  static const Int32 __MAX_CELL_EDGE = 12;
#endif

  //! Nombre de type d'éléments
  static const Integer NB_TYPE = NB_BASIC_ITEM_TYPE;

  //! Nombre maximum de noeuds par maille
  static const Integer MAX_CELL_NODE = __MAX_CELL_NODE;

  //! Nombre maximum de sommets par face
  static const Integer MAX_FACE_NODE = __MAX_FACE_NODE;

  //! Nombre maximum de faces par maille
  static const Integer MAX_CELL_FACE = __MAX_CELL_FACE;

  //! Nombre maximum d'arêtes par maille
  static const Integer MAX_CELL_EDGE = __MAX_CELL_EDGE;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
