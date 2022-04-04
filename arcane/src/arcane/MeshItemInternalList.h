// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshItemInternalList.h                                      (C) 2000-2016 */
/*                                                                           */
/* Tableaux d'indirection sur les entités d'un maillage.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHITEMINTERNALLIST_H
#define ARCANE_MESHITEMINTERNALLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tableaux d'indirection sur les entités d'un maillage.
 */
class MeshItemInternalList
{
 public:

  MeshItemInternalList() : mesh(nullptr) {}

 public:

  ItemInternalArrayView nodes;
  ItemInternalArrayView edges;
  ItemInternalArrayView faces;
  ItemInternalArrayView cells;
  ItemInternalArrayView dualNodes;
  ItemInternalArrayView links;
  ItemInternalArrayView particles;
  IMesh* mesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

