// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedFace.h                                                  (C) 2000-2016 */
/*                                                                           */
/* Face semi-conforme du maillage.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TIEDFACE_H
#define ARCANE_TIEDFACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Face semi-conforme du maillage.
 */
class TiedFace
{
 public:

 public:

  TiedFace(Integer aindex,Face aface)
  : m_index(aindex), m_face(aface)
  {
  }

  TiedFace()
  : m_index(NULL_ITEM_ID)
  {
  }

 public:

  //! Indice de la face dans la liste des faces soudées de la face maitre
  Integer index() const { return m_index; }

  //! Face soudée
  Face face() const { return m_face; }

 private:

  //! Indice de la face dans la liste des faces soudées de la face maitre
  Integer m_index;
  //! Face soudée
  Face m_face;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

