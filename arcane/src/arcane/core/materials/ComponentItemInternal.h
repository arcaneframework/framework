// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternal.h                                     (C) 2000-2023 */
/*                                                                           */
/* Partie interne d'une maille multi-matériau.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEMINTERNAL_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEMINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInternal.h"
#include "arcane/Item.h"
#include "arcane/core/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ComponentItemSharedInfo
{
 public:
  Int32* m_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne d'une maille matériau ou milieu.
 *
 * Cette classe est le pendant de ItemInternal pour la gestion des matériaux
 * et des milieux. Elle ne doit en principe pas être utilisée directement, sauf
 * par les classes de Arcane. Il vaut mieux utiliser les
 * classes ComponentCell,  MatCell, EnvCell ou AllEnvCell.
 *
 * \todo pour économiser la mémoire, utiliser un ComponentItemSharedInfo
 * pour stocker une fois les infos multiples.
 */
class ARCANE_CORE_EXPORT ComponentItemInternal
{
 private:

  static const int MAT_INDEX_OFFSET = 10;

  //! Entité nulle
  static ComponentItemInternal nullComponentItemInternal;

 public:

  //! Entité nulle
  static ComponentItemInternal* nullItem() { return &nullComponentItemInternal; }

 public:

  ComponentItemInternal()
  : m_component_id(-1), m_level(-1), m_nb_sub_component_item(0), m_component(0),
    m_super_component_item(0), m_first_sub_component_item(0), m_global_item(ItemInternal::nullItem())
  {
    m_var_index.reset();
  }

 public:

  /*!
   * \internal
   * Positionne l'indexeur dans les variables matériaux.
   */
  void setVariableIndex(MatVarIndex index)
  {
    m_var_index = index;
  }

  //! Indexeur dans les variables matériaux
  MatVarIndex variableIndex() const
  {
    return m_var_index;
  }

  //! Identifiant du composant
  Int32 componentId() const { return m_component_id; }

  //! Indique s'il s'agit de la maille nulle.
  bool null() const { return m_var_index.null(); }

  //! Composant
  IMeshComponent* component() const { return m_component; }

  //! Composant supérieur (0 si aucun)
  ComponentItemInternal* superItem() const { return m_super_component_item; }

  void setSuperAndGlobalItem(ComponentItemInternal* cii,Item ii)
  {
    m_super_component_item = cii;
    m_global_item = ItemCompatibility::_itemInternal(ii);
  }

  void setGlobalItem(Item ii)
  {
    m_global_item = ItemCompatibility::_itemInternal(ii);
  }

  //! Nombre de sous-composants.
  Int32 nbSubItem() const
  {
    return m_nb_sub_component_item;
  }

  //! Première entité sous-composant.
  ComponentItemInternal* firstSubItem() const { return m_first_sub_component_item; }

  //! Entité globale correspondante.
  impl::ItemBase globalItemBase() { return m_global_item; }

  //! Positionne le nombre de sous-composants.
  void setNbSubItem(Int32 nb_sub_item)
  {
    m_nb_sub_component_item = nb_sub_item;
  }

  //! Positionne le premier sous-composant.
  void setFirstSubItem(ComponentItemInternal* first_sub_item)
  {
    m_first_sub_component_item = first_sub_item;
  }

  void setComponent(IMeshComponent* component,Int32 component_id)
  {
    m_component = component;
#ifdef ARCANE_CHECK
    _checkIsInt16(component_id);
#endif
    m_component_id = (Int16)component_id;
  }

  Int32 level() const { return m_level; }

  void setLevel(Int32 level)
  {
#ifdef ARCANE_CHECK
    _checkIsInt16(level);
#endif
    m_level = (Int16)level;
  }

  //! Numéro unique de l'entité component
  Int64 componentUniqueId() const
  {
    // TODO: Vérifier que arrayIndex() ne dépasse pas (1<<MAT_INDEX_OFFSET)
    return (Int64)m_var_index.arrayIndex() + ( (Int64)m_global_item->uniqueId() << MAT_INDEX_OFFSET );
  }

  void reset()
  {
    m_var_index.reset();
    m_component_id = -1;
    m_component = 0;
    m_level = (-1);
    m_super_component_item = 0;
    m_nb_sub_component_item = 0;
    m_first_sub_component_item = 0;
    m_global_item = ItemInternal::nullItem();
  }

 protected:

  MatVarIndex m_var_index;
  Int16 m_component_id;
  Int16 m_level;
  Int32 m_nb_sub_component_item;
  IMeshComponent* m_component;
  ComponentItemInternal* m_super_component_item;
  ComponentItemInternal* m_first_sub_component_item;
  ItemInternal* m_global_item;

 private:

  void _checkIsInt16(Int32 v)
  {
    if (v<(-32768) || v>32767)
      _throwBadCast(v);
  }
  void _throwBadCast(Int32 v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
