// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroupBuilder.h                                      (C) 2000-2021 */
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

namespace Arcane
{

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
  explicit ItemPairGroupBuilder(const ItemPairGroup& group);
  ~ItemPairGroupBuilder();

 public:

  //! groupe associé.
  const ItemPairGroup& group() { return m_group; }
  //! Ajoute les entités \a sub_items a
  void addNextItem(Int32ConstArrayView sub_items);

 private:

  ItemPairGroup m_group;
  Int64 m_index;
  Array<Int64>& m_unguarded_indexes;
  Array<Int32>& m_unguarded_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
