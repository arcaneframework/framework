// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedFace.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Face semi-conforme du maillage.                                           */
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
 * \brief Face semi-conforme du maillage.
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

  //! Indice de la face dans la liste des faces soudées de la face maitre
  Integer index() const { return m_index; }

  //! Face soudée
  Face face() const { return m_face; }

 private:

  //! Indice de la face dans la liste des faces soudées de la face maitre
  Integer m_index = NULL_ITEM_LOCAL_ID;
  //! Face soudée
  Face m_face;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

