// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItem.h                                           (C) 2000-2024 */
/*                                                                           */
/* Entité représentant un constituant d'une maille multi-matériaux.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CONSTITUENTITEM_H
#define ARCANE_CORE_MATERIALS_CONSTITUENTITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemInternal.h"

#include "arcane/core/Item.h"

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
 * possible de récupérer le ConstituentItem de niveau supérieur via
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
class ARCANE_CORE_EXPORT ConstituentItem
{
  // Pour accéder à _internal()
  friend CellComponentCellEnumerator;
  friend EnvCellVector;
  friend MatCellVector;
  friend MeshMaterialMng;
  friend AllEnvCell;
  friend EnvCell;
  friend MatCell;
  friend ConstituentItemVectorImpl;

 public:

  ARCCORE_HOST_DEVICE ConstituentItem(const matimpl::ConstituentItemBase& mii)
  : m_constituent_item_index(mii.m_constituent_item_index)
  , m_shared_info(mii.m_shared_info)
  {}

  ARCCORE_HOST_DEVICE ConstituentItem()
  {
#ifndef ARCCORE_DEVICE_CODE
    m_shared_info = ComponentItemSharedInfo::_nullInstance();
#endif
  }

 public:

  //! Opérateur de conversion vers un ComponentItemLocalId
  ARCCORE_HOST_DEVICE operator ComponentItemLocalId() const { return ComponentItemLocalId{ _varIndex() }; }

 public:

  //! \internal
  ARCCORE_HOST_DEVICE MatVarIndex _varIndex() const { return m_shared_info->_varIndex(m_constituent_item_index); }

  ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase constituentItemBase() const { return { m_shared_info, m_constituent_item_index }; }

  /*!
   * \brief Constituant associé.
   * \pre null()==false
   */
  IMeshComponent* component() const { return m_shared_info->_component(m_constituent_item_index); }

  //! Identifiant du composant dans la liste des composants de ce type.
  ARCCORE_HOST_DEVICE Int32 componentId() const { return m_shared_info->_componentId(m_constituent_item_index); }

  //! Indique s'il s'agit de la maille nulle
  ARCCORE_HOST_DEVICE bool null() const { return m_constituent_item_index.isNull(); }

  //! Maille de niveau supérieur dans la hiérarchie
  ARCCORE_HOST_DEVICE ComponentCell superCell() const { return ComponentCell(_superItemBase()); }

  //! Niveau hiérarchique de l'entité
  ARCCORE_HOST_DEVICE Int32 level() const { return m_shared_info->m_level; }

  //! Nombre de sous-éléments
  ARCCORE_HOST_DEVICE Int32 nbSubItem() const { return m_shared_info->_nbSubConstituent(m_constituent_item_index); }

  //! Maille globale
  Cell globalCell() const
  {
    return Cell(m_shared_info->_globalItemBase(m_constituent_item_index));
  }

  //! localId() de la maille globale
  ARCCORE_HOST_DEVICE CellLocalId globalCellId() const
  {
    return CellLocalId(m_shared_info->_globalItemId(m_constituent_item_index));
  }

  /*!
   * \brief Numéro unique de l'entité constituant.
   *
   * Ce numéro est unique pour chaque constituant de chaque maille.
   *
   * \warning Ce numéro unique n'est pas le même que celui de la maille globale
   * associée.
   */
  Int64 componentUniqueId() const { return m_shared_info->_componentUniqueId(m_constituent_item_index); }

  //! Liste des sous-constituents de cette entité
  ARCCORE_HOST_DEVICE inline CellComponentCellEnumerator subItems() const;

  //! Opérateur d'écriture
  ARCANE_CORE_EXPORT friend std::ostream&
  operator<<(std::ostream& o, const ComponentCell& mvi);

 protected:

  static ARCCORE_HOST_DEVICE void _checkLevel([[maybe_unused]] matimpl::ConstituentItemBase item_base,
                                              [[maybe_unused]] Int32 expected_level)
  {
#if !defined(ARCCORE_DEVICE_CODE)
    if (item_base.null())
      return;
    Int32 lvl = item_base.level();
    if (lvl != expected_level)
      _badConversion(item_base, lvl, expected_level);
#endif
  }
  static void _badConversion(matimpl::ConstituentItemBase item_base, Int32 level, Int32 expected_level);

  ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(Int32 index) const
  {
    return m_shared_info->_subItemBase(m_constituent_item_index, index);
  }
  ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _superItemBase() const
  {
    return m_shared_info->_superItemBase(m_constituent_item_index);
  }
  ARCCORE_HOST_DEVICE ConstituentItemIndex _firstSubConstituentLocalId() const
  {
    return m_shared_info->_firstSubConstituentLocalId(m_constituent_item_index);
  }

 protected:

  ConstituentItemIndex m_constituent_item_index;
  ComponentItemSharedInfo* m_shared_info = nullptr;

 private:

  //! \internal
  ARCCORE_HOST_DEVICE ConstituentItemIndex _constituentItemIndex() const
  {
    return m_constituent_item_index;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur les constituants d'une maille.
 */
class ARCANE_CORE_EXPORT CellComponentCellEnumerator
{
  friend class EnumeratorTracer;

 public:

  class Sentinel
  {};
  class Iterator
  {
    friend class CellComponentCellEnumerator;

   public:

    using iterator_category = std::forward_iterator_tag;
    using value_type = ComponentCell;

   private:

    ARCCORE_HOST_DEVICE explicit Iterator(CellComponentCellEnumerator enumerator)
    : m_index(enumerator.m_index)
    , m_size(enumerator.m_size)
    , m_first_sub_index(enumerator.m_first_sub_index)
    , m_sub_constituent_shared_info(enumerator.m_sub_constituent_shared_info)
    {}

   public:

    ARCCORE_HOST_DEVICE void operator++() { ++m_index; }
    ARCCORE_HOST_DEVICE ComponentCell operator*() const
    {
      ARCANE_CHECK_AT(m_index, m_size);
      return ComponentCell(_currentSubItemBase());
    }
    ARCCORE_HOST_DEVICE operator ComponentItemLocalId() const
    {
      return ComponentItemLocalId(_varIndex());
    }
    friend ARCCORE_HOST_DEVICE bool operator==(const Iterator& x, const Sentinel&)
    {
      return x.m_index == x.m_size;
    }

   private:

    Int32 m_index = 0;
    Int32 m_size = 0;
    Int32 m_first_sub_index = -1;
    ComponentItemSharedInfo* m_sub_constituent_shared_info = nullptr;

   private:

    ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _currentSubItemBase() const
    {
      return m_sub_constituent_shared_info->_item(ConstituentItemIndex(m_first_sub_index + m_index));
    }
    ARCCORE_HOST_DEVICE MatVarIndex _varIndex() const
    {
      return m_sub_constituent_shared_info->_varIndex(ConstituentItemIndex(m_first_sub_index + m_index));
    }
  };

  template <typename ConstituentItemType>
  class IteratorT : public Iterator
  {
   public:

    using value_type = ConstituentItemType;
    friend class CellComponentCellEnumeratorT<ConstituentItemType>;

   private:

    ARCCORE_HOST_DEVICE explicit IteratorT(CellComponentCellEnumeratorT<ConstituentItemType> enumerator)
    : Iterator(enumerator)
    {}

   public:

    ARCCORE_HOST_DEVICE ConstituentItemType operator*() const
    {
      ARCANE_CHECK_AT(m_index, m_size);
      return ConstituentItemType(_currentSubItemBase());
    }
  };

 public:

  ARCCORE_HOST_DEVICE explicit CellComponentCellEnumerator(ComponentCell super_item)
  : m_size(super_item.nbSubItem())
  , m_first_sub_index(super_item._firstSubConstituentLocalId().localId())
  , m_sub_constituent_shared_info(super_item.m_shared_info->m_sub_component_item_shared_info)
  {
  }

 public:

  ARCCORE_HOST_DEVICE void operator++() { ++m_index; }
  ARCCORE_HOST_DEVICE bool hasNext() const { return m_index < m_size; }

  ARCCORE_HOST_DEVICE ComponentCell operator*() const
  {
    ARCANE_CHECK_AT(m_index, m_size);
    return ComponentCell(_currentSubItemBase());
  }
  ARCCORE_HOST_DEVICE MatVarIndex _varIndex() const
  {
    return m_sub_constituent_shared_info->_varIndex(ConstituentItemIndex(m_first_sub_index + m_index));
  }
  ARCCORE_HOST_DEVICE Int32 index() const { return m_index; }
  ARCCORE_HOST_DEVICE operator ComponentItemLocalId() const
  {
    return ComponentItemLocalId(_varIndex());
  }
  ARCCORE_HOST_DEVICE Iterator begin() const { return Iterator(*this); }
  ARCCORE_HOST_DEVICE Sentinel end() const { return {}; }

 protected:

  Int32 m_index = 0;
  Int32 m_size = 0;
  Int32 m_first_sub_index = -1;
  ComponentItemSharedInfo* m_sub_constituent_shared_info = nullptr;

 protected:

  ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _currentSubItemBase() const
  {
    return m_sub_constituent_shared_info->_item(ConstituentItemIndex(m_first_sub_index + m_index));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur typés sur les mailles composants d'une maille.
 */
template <typename ComponentCellType> class CellComponentCellEnumeratorT
: public CellComponentCellEnumerator
{
 public:

  using IteratorType = CellComponentCellEnumerator::IteratorT<ComponentCellType>;

 public:

  explicit ARCCORE_HOST_DEVICE CellComponentCellEnumeratorT(ComponentCell super_item)
  : CellComponentCellEnumerator(super_item)
  {}

 public:

  ARCCORE_HOST_DEVICE ComponentCellType operator*() const
  {
    ARCANE_CHECK_AT(m_index, m_size);
    return ComponentCellType(_currentSubItemBase());
  }
  ARCCORE_HOST_DEVICE IteratorType begin() const { return IteratorType(*this); }
  ARCCORE_HOST_DEVICE Sentinel end() const { return {}; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_HOST_DEVICE CellComponentCellEnumerator ComponentCell::
subItems() const
{
  return CellComponentCellEnumerator(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
