// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFunctor.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Fonctor sur les entités.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemFunctor.h"
#include "arcane/utils/Math.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractItemRangeFunctor::
AbstractItemRangeFunctor(ItemVectorView items_view,Integer grain_size)
: m_items(items_view)
, m_block_size(SIMD_PADDING_SIZE)
, m_nb_block(items_view.size())
, m_block_grain_size(grain_size)
{
  // NOTE: si le range functor est utilisé pour la vectorisation, il faut
  // que items_view.localIds() soit aligné. Le problème est qu'on ne sait
  // pas exactement quel est l'alignement requis. On pourrait se base sur
  // \a m_block_size et dire que l'alignement est m_block_size * sizeof(Int32).
  // De toute facon, le problème éventuel d'alignement sera détecté par
  // SimdItemEnumerator.
  Integer nb_item = m_items.size();
  m_nb_block = nb_item / m_block_size;
  if ( (nb_item % m_block_size)!=0 )
    ++m_nb_block;

  m_block_grain_size = grain_size / m_block_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView AbstractItemRangeFunctor::
_view(Integer begin_block, Integer nb_block, Int32* true_begin) const
{
  // Converti (begin_block,nb_block) en (begin,size) correspondant à m_items.
  Integer begin = begin_block * m_block_size;
  Integer nb_item = m_items.size();
  Integer size = math::min(nb_block * m_block_size,nb_item-begin);
  if (true_begin)
    *true_begin = begin;
  return m_items.subView(begin,size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
