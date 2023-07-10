// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItem.h                                             (C) 2000-2023 */
/*                                                                           */
/* Entité composant d'une maillage multi-matériau.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEM_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemInternal.h"

#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Représente un composant d'une maille multi-matériau.
 *
 * Cet objet représente un composant d'une maille multi-matériau. Par
 * composant, on entend un matériau (MatCell), un milieu (EnvCell) ou
 * un allenvcell (AllEnvCell).
 *
 * Il existe une notion de hiérarchie entre ces composants et il est
 * possible de récupérer la ComponentCell de niveau supérieur via
 * superCell(). Pour itérer sur les éléments de niveau inférieur, il
 * est possible d'utiliser la macro ENUMERATE_CELL_COMPONENTCELL()
 *
 * Il existe une maille spéciale, appelée maille nulle, pour laquelle
 * null() est vrai et qui représente une maille invalide. Dans le
 * cas de la maille invalide, il ne faut appeler aucune des autres
 * méthode de la classe sous peine de provoquer un plantage.
 *
 * \warning Ces mailles sont invalidées dès que la liste des mailles d'un
 * matériau ou d'un milieux change. Il ne faut donc pas
 * conserver une maille de ce type entre deux changements de cette liste.
 */
class ARCANE_CORE_EXPORT ComponentCell
{
 public:

  ComponentCell(ComponentItemInternal* mii) : m_internal(mii){}
  ComponentCell() : m_internal(ComponentItemInternal::nullItem()){}

 public:

  //! Opérateur de conversion vers un ComponentItemLocalId
  operator ComponentItemLocalId() const { return ComponentItemLocalId{ _varIndex() }; }

 public:

  //! \internal
  MatVarIndex _varIndex() const { return m_internal->variableIndex(); }

  //! \internal
  ComponentItemInternal* internal() const { return m_internal; }

  //! Composant associé
  IMeshComponent* component() const { return m_internal->component(); }

  //! Identifiant du composant dans la liste des composants de ce type.
  Int32 componentId() const { return m_internal->componentId(); }

  //! Indique s'il s'agit de la maille nulle
  bool null() const { return m_internal->null(); }

  //! Maille de niveau supérieur dans la hiérarchie
  ComponentCell superCell() const { return m_internal->superItem(); }

  //! Niveau hiérarchique de l'entité
  Int32 level() const { return m_internal->level(); }

  //! Nombre de sous-éléments
  Int32 nbSubItem() const { return m_internal->nbSubItem(); }

  //! Maille arcane
  Cell globalCell() const
  {
    return Cell(m_internal->globalItemBase());
  }

  /*!
   * \brief Numéro unique de l'entité constituant.
   *
   * Ce numéro est unique pour chaque constituant de chaque maille.
   *
   * \warning Ce numéro unique n'est pas le même que celui de la maille globale
   * associée.
   */
  Int64 componentUniqueId() const { return m_internal->componentUniqueId(); }

 protected:

  static void _checkLevel(ComponentItemInternal* internal,Int32 expected_level)
  {
    if (internal->null())
      return;
    Int32 lvl = internal->level();
    if (!internal->null() && lvl!=expected_level)
      _badConversion(lvl,expected_level);
  }
  static void _badConversion(Int32 level,Int32 expected_level);

 protected:

  ComponentItemInternal* m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

