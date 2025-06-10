// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroupImpl.h                                         (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'un tableau de listes d'entités.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMPAIRGROUPIMPL_H
#define ARCANE_CORE_ITEMPAIRGROUPIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/SharedReference.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemPairGroupImplPrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'un tableau de listes d'entités.
 */
class ARCANE_CORE_EXPORT ItemPairGroupImpl
: public SharedReference
{
 public:

  ItemPairGroupImpl();
  ItemPairGroupImpl(const ItemGroup& group, const ItemGroup& sub_group);
  ~ItemPairGroupImpl() override; //!< Libère les ressources

 public:

  static ItemPairGroupImpl* shared_null;
  static ItemPairGroupImpl* checkSharedNull();

 public:

  virtual ISharedReference& sharedReference() { return *this; }

 public:

  //! Nom du groupe
  const String& name() const;

  //! Nombre de références sur le groupe.
  virtual Integer nbRef() const { return refCount(); }

  //! Retourne \a true si le groupe est nul.
  bool null() const;

  //! Maillage auquel appartient le groupe (0 pour le groupe nul).
  IMesh* mesh() const;

  //! Genre du groupe. Il s'agit du genre de ses éléments.
  eItemKind itemKind() const;

  //! Genre du groupe. Il s'agit du genre de ses éléments.
  eItemKind subItemKind() const;

  //! Famille à laquelle appartient le groupe (ou 0 si aucune)
  IItemFamily* itemFamily() const;

  //! Famille à laquelle appartient le groupe (ou 0 si aucune)
  IItemFamily* subItemFamily() const;

  //! Groupe des entités
  const ItemGroup& itemGroup() const;

  //! Groupe des sous-entités
  const ItemGroup& subItemGroup() const;

  //! Nombre d'entités du groupe
  Integer size() const;

  //! Invalide le groupe
  void invalidate(bool force_recompute);

  //! Vérifie que le groupe est valide.
  void checkValid();

  /*!
   * \brief Réactualise le groupe si nécessaire.
   *
   * Un groupe doit être réactualisée lorsqu'il est devenu invalide, par exemple
   * suite à un appel à invalidate().
   * \retval true si le groupe a été réactualisé,
   * \retval false sinon.
   */
  bool checkNeedUpdate();

  //! Change les indices des entités du groupe
  void changeIds(IntegerConstArrayView old_to_new_ids);

  /*!
   * \internal
   */
  Array<Int64>& unguardedIndexes() const;

  /*!
   * \internal
   */
  Array<Int32>& unguardedLocalIds() const;

  /*!
   * \internal
   */
  void setComputeFunctor(IFunctor* functor);

  Int64ArrayView indexes();

  Span<const Int32> subItemsLocalId();

 private:

  ItemPairGroupImplPrivate* m_p = nullptr; //!< Implémentation du groupe

 public:

  void addRef() override;
  void removeRef() override;

 private:

  void deleteMe() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
