// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroup.h                                             (C) 2000-2025 */
/*                                                                           */
/* Tableau de listes d'entités.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMPAIRGROUP_H
#define ARCANE_CORE_ITEMPAIRGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/AutoRef.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/IFunctorWithArgument.h"

#include "arcane/core/ItemPairGroupImpl.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//NOTE: la documentation complète est dans ItemPairGroup.cc
/*!
 * \brief Tableau de listes d'entités.
 */
class ARCANE_CORE_EXPORT ItemPairGroup
{
 public:

  /*!
   * \brief Functor pour un calcul personnalisé des connectivités.
   */
  typedef IFunctorWithArgumentT<ItemPairGroupBuilder&> CustomFunctor;
  class CustomFunctorWrapper;

 public:

  //! Construit un tableau vide.
  ItemPairGroup();
  //! Construit un groupe à partir de la représentation interne \a prv
  explicit ItemPairGroup(ItemPairGroupImpl* prv);
  /*!
   * \brief Construit une instance en spécifiant le voisinage via les entités
   * de genre \a link_kind.
   */
  ItemPairGroup(const ItemGroup& group, const ItemGroup& sub_item_group,
                eItemKind link_kind);
  //! Construit une instance avec un fonctor particulier.
  ItemPairGroup(const ItemGroup& group, const ItemGroup& sub_item_group,
                CustomFunctor* functor);
  //! Constructeur de recopie.
  ItemPairGroup(const ItemPairGroup& from)
  : m_impl(from.m_impl)
  {}

  const ItemPairGroup& operator=(const ItemPairGroup& from)
  {
    m_impl = from.m_impl;
    return (*this);
  }
  virtual ~ItemPairGroup() = default;

 public:

  //! \a true is le groupe est le groupe nul
  inline bool null() const { return m_impl->null(); }
  //! Type des entités du groupe
  inline eItemKind itemKind() const { return m_impl->itemKind(); }
  //! Type des sous-entités du groupe
  inline eItemKind subItemKind() const { return m_impl->subItemKind(); }

 public:

  /*!
   * \brief Retourne l'implémentation du groupe.
   *
   * \warning Cette méthode retourne un pointeur sur la représentation
   * interne du groupe et ne doit pas être utilisée
   * en dehors d'Arcane.
   */
  ItemPairGroupImpl* internal() const { return m_impl.get(); }

  //! Famille d'entité à laquelle appartient ce groupe (0 pour une liste nulle)
  IItemFamily* itemFamily() const { return m_impl->itemFamily(); }

  //! Famille d'entité à laquelle appartient ce groupe (0 pour une liste nulle)
  IItemFamily* subItemFamily() const { return m_impl->subItemFamily(); }

  //! Maillage auquel appartient cette liste (0 pour une liste nulle)
  IMesh* mesh() const { return m_impl->mesh(); }

  //! Groupe des items initiaux
  const ItemGroup& itemGroup() const { return m_impl->itemGroup(); }

  //! Groupe des items finaux (après rebond)
  const ItemGroup& subItemGroup() const { return m_impl->subItemGroup(); }

 public:

  /*! \brief Invalide la liste.
   */
  void invalidate(bool force_recompute = false)
  {
    m_impl->invalidate(force_recompute);
  }

  //! Vérification interne de la validité du groupe
  void checkValid() { m_impl->checkValid(); }

 public:

  ItemPairEnumerator enumerator() const;

 protected:

  //! Représentation interne du groupe.
  AutoRefT<ItemPairGroupImpl> m_impl;

 protected:

  //! Retourne le groupe \a impl s'il est du genre \a kt, le groupe nul sinon
  static ItemPairGroupImpl* _check(ItemPairGroupImpl* impl, eItemKind ik, eItemKind aik)
  {
    return (impl->itemKind() == ik && impl->subItemKind() == aik) ? impl : ItemPairGroupImpl::checkSharedNull();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare les références de deux groupes.
 * \retval true si \a g1 et \a g2 réfèrent au même groupe,
 * \retval false sinon.
 */
inline bool
operator==(const ItemPairGroup& g1, const ItemPairGroup& g2)
{
  return g1.internal() == g2.internal();
}

/*!
 * \brief Compare les références de deux groupes.
 * \retval true si \a g1 et \a g2 ne réfèrent pas au même groupe,
 * \retval false sinon.
 */
inline bool
operator!=(const ItemPairGroup& g1, const ItemPairGroup& g2)
{
  return g1.internal() != g2.internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence à un groupe d'un genre donné.
 */
template <typename ItemKind, typename SubItemKind>
class ItemPairGroupT
: public ItemPairGroup
{
 public:

  //! Type de cette classe
  typedef ItemPairGroupT<ItemKind, SubItemKind> ThatClass;
  //! Type de la classe contenant les caractéristiques de l'entité
  typedef ItemTraitsT<ItemKind> TraitsType;
  typedef ItemTraitsT<SubItemKind> SubTraitsType;

  typedef typename TraitsType::ItemType ItemType;
  typedef typename TraitsType::ItemGroupType ItemGroupType;
  typedef typename SubTraitsType::ItemType SubItemType;
  typedef typename SubTraitsType::ItemGroupType SubItemGroupType;

 public:

  ItemPairGroupT() {}
  ItemPairGroupT(const ItemPairGroup& from)
  : ItemPairGroup(_check(from.internal(), TraitsType::kind(), SubTraitsType::kind()))
  {}
  ItemPairGroupT(const ThatClass& from)
  : ItemPairGroup(from)
  {}
  ItemPairGroupT(const ItemGroupType& group, const SubItemGroupType& sub_group,
                 eItemKind link_kind)
  : ItemPairGroup(group, sub_group, link_kind)
  {}
  ItemPairGroupT(const ItemGroupType& group, const SubItemGroupType& sub_group,
                 CustomFunctor* functor)
  : ItemPairGroup(group, sub_group, functor)
  {}
  ~ItemPairGroupT() {}

 public:

  const ThatClass& operator=(const ThatClass& from)
  {
    m_impl = from.internal();
    return (*this);
  }
  const ThatClass& operator=(const ItemPairGroup& from)
  {
    _assign(from);
    return (*this);
  }

 protected:

  void _assign(const ItemPairGroup& from)
  {
    m_impl = _check(from.internal(), TraitsType::kind(), SubTraitsType::kind());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
