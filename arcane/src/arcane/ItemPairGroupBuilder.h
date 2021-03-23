// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroupBuilder.h                                      (C) 2000-2016 */
/*                                                                           */
/* Construction des listes des entités des ItemPairGroup.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMPAIRGROUPBUILDER_H
#define ARCANE_ITEMPAIRGROUPBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemPairGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des listes des entités des ItemPairGroup.
 *
 * Cette classe est utilisée lors du recalcul des entités d'un ItemPairGroup.
 *
 * Le code utilisateur doit appeler la méthode addNextItem() pour chaque entité
 * de group().itemGroup() en spécifiant les localId() des entités ajoutées.
 * Par exemple:
 *
 \code
 * void functor(ItemPairGroupBuilder& builder)
 * {
 *    Int32Array local_ids;
 *    ENUMERATE_ITEM(iitem.builder.group().itemGroup()){
 *      local_ids.clear();
 *      // Calcule les entité connectées à \a iitem et les ajoute à \a local_ids.
 *      ...
 *      builder.addNextItem(local_ids);
 *    }
 * }
 \endcode
 *
 * Pour un exemple plus complet d'utilisation, se référer à la documentation
 * de ItemPairGroup.
 */
class ARCANE_CORE_EXPORT ItemPairGroupBuilder
{
 public:

  //! \internal
  ItemPairGroupBuilder(const ItemPairGroup& group);
  ~ItemPairGroupBuilder();

 public:

  //! groupe associé.
  const ItemPairGroup& group() { return m_group; }
  //! Ajoute les entités \a sub_items a
  void addNextItem(Int32ConstArrayView sub_items);

 private:

  ItemPairGroup m_group;
  Integer m_index;
  IntegerArray& m_unguarded_indexes;
  Int32Array& m_unguarded_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
