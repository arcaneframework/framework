// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonItemGroupFilterer.h                                   (C) 2000-2021 */
/*                                                                           */
/* Filtrage des groupes communs à toutes les parties d'un maillage.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_COMMONITEMGROUPFILTERINFO_H
#define ARCANE_COMMONITEMGROUPFILTERINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe utilitaire des groupes communs à toutes les parties
 * d'un maillage.
 *
 * Pour utiliser cette classe, il faut ajouter via addGroupToFilter()
 * les groupes qu'on souhaite filtrer. Il faut ensuite appeler la
 * méthode applyFiltering() pour effectuer le filtrage.
 */
class ARCANE_MESH_EXPORT CommonItemGroupFilterer
{
 public:

  explicit CommonItemGroupFilterer(IItemFamily* family);

 public:

  CommonItemGroupFilterer(const CommonItemGroupFilterer& rhs) = delete;
  CommonItemGroupFilterer& operator=(const CommonItemGroupFilterer& rhs) = delete;

 public:

  void addGroupToFilter(const ItemGroup& group);
  /*!
   * \brief Filtre les groupes communs.
   *
   * S'il existe des groupes qui ne sont pas communs à tous les rangs, une
   * exception est levée.
   */
  void applyFiltering();

  //! Liste triée par ordre alphabétique des groupes communs.
  ItemGroupCollection sortedCommonGroups() { return m_sorted_common_groups; }

 private:

  IItemFamily* m_family;
  List<ItemGroup> m_input_groups;
  List<ItemGroup> m_sorted_common_groups;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
