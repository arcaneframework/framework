// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemIndexedSelectionView.h                       (C) 2000-2026 */
/*                                                                           */
/* Vue sur un sous ensemble d'un conteneur de ConstituentItem.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CONSTITUENTITEMINDEXEDSELECTIONVIEW_H
#define ARCANE_CORE_MATERIALS_CONSTITUENTITEMINDEXEDSELECTIONVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/ComponentItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials::Impl
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques pour le conteneur associé à
 * ConstituentItemIndexedSelectionView.
 *
 * Cette classe doit être spécialisée. Elle l'est pour les conteneurs
 * de type ComponentItemVectorView, MatCellVectorView, EnvCellVectorView
 * ou SmallSpan<T>, avec T de type ComponentCell, MatCell ou EnvCell.
 */
template <typename ContainerType_>
struct ConstituentItemIndexedSelectionViewTraits;

template <typename ConstituentContainerType_>
struct ConstituentItemIndexedSelectionViewTraitsBase
{
  using ThatContainer = ConstituentContainerType_;
  using ValueType = ThatContainer::ValueType;
  static constexpr bool IsSpan() { return false; }
  static Int32 size(ThatContainer v)
  {
    return v.nbItem();
  }
};

template <>
struct ConstituentItemIndexedSelectionViewTraits<ComponentItemVectorView>
: ConstituentItemIndexedSelectionViewTraitsBase<ComponentItemVectorView>
{
  static ARCCORE_HOST_DEVICE ComponentCell item(ComponentItemVectorView v, Int32 i)
  {
    return v.componentCell(i);
  }
};

template <>
struct ConstituentItemIndexedSelectionViewTraits<MatCellVectorView>
: ConstituentItemIndexedSelectionViewTraitsBase<MatCellVectorView>
{
  static ARCCORE_HOST_DEVICE MatCell item(MatCellVectorView v, Int32 i)
  {
    return v.matCell(i);
  }
};

template <>
struct ConstituentItemIndexedSelectionViewTraits<EnvCellVectorView>
: ConstituentItemIndexedSelectionViewTraitsBase<EnvCellVectorView>
{
  static ARCCORE_HOST_DEVICE EnvCell item(EnvCellVectorView v, Int32 i)
  {
    return v.envCell(i);
  }
};

//! Spécialisation partielle pour un SmallSpan<T>
template <typename ConstituentItemType_>
class ConstituentItemIndexedSelectionViewTraitsSpanBase
{
  using ThatContainer = SmallSpan<const ConstituentItemType_>;
  using ValueType = ConstituentItemType_;
  static constexpr bool IsSpan() { return true; }
  static ARCCORE_HOST_DEVICE Int32 size(ThatContainer v)
  {
    return v.size();
  }
  static ARCCORE_HOST_DEVICE ValueType item(ThatContainer v, Int32 i)
  {
    return v[i];
  }
};

template <>
struct ConstituentItemIndexedSelectionViewTraits<SmallSpan<const ConstituentItem>>
: ConstituentItemIndexedSelectionViewTraitsSpanBase<ConstituentItem>
{
};
template <>
struct ConstituentItemIndexedSelectionViewTraits<SmallSpan<const MatCell>>
: ConstituentItemIndexedSelectionViewTraitsSpanBase<MatCell>
{
};
template <>
struct ConstituentItemIndexedSelectionViewTraits<SmallSpan<const EnvCell>>
: ConstituentItemIndexedSelectionViewTraitsSpanBase<EnvCell>
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour ConstituentItemIndexedSelectionView.
 */
class ARCANE_CORE_EXPORT ConstituentItemIndexedSelectionViewBase
{
 public:

  using IndexArrayView = const SmallSpan<const Int32>;

 protected:

  explicit ConstituentItemIndexedSelectionViewBase(SmallSpan<const Int32> indices);
  explicit ConstituentItemIndexedSelectionViewBase(IMeshComponent* constituent, Int32 selection_size);

 public:

  //! nombre de EnvCell sélectionnées
  ARCCORE_HOST_DEVICE Int32 size() const { return m_selection_view.size(); }

 protected:

  /*!
   * \brief Sélection.
   *
   * Si ce champ est omis à la construction, le défaut sera une sélection 'pleine'
   * (i.e. tous les éléments d'origine, dans le même ordre)
   */
  SmallSpan<const Int32> m_selection_view = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un sous ensemble d'un conteneur de ConstituentItem.
 *
 * Le conteneur est l'argument template \a ContainerView_. Il peut
 * s'agit d'un ComponentItemVectorView, MatCellVectorView, EnvCellVectorView
 * ou juste d'un SmallSpan d'un ConstituentItem.
 * La sélection des entités se fait par un tableau d'indices.
 * Si ce tableau n'est pas fourni, la sélection est sur l'ensemble des entités.
 *
 * Comme toute vue, les instances de cette classe sont invalidées si les
 * constituants évoluent (ajout ou suppression)
 */
template <typename ContainerView_>
class ConstituentItemIndexedSelectionView
: public ConstituentItemIndexedSelectionViewBase
{
 public:

  using ItemVecView = ContainerView_;
  using ThatClass = ConstituentItemIndexedSelectionView;
  using TraitsType = Impl::ConstituentItemIndexedSelectionViewTraits<ContainerView_>;
  using ValueType = TraitsType::ValueType;
  static constexpr bool IsSpanContainer() { return TraitsType::IsSpan(); }

 public:

  ConstituentItemIndexedSelectionView(ItemVecView ecv, IndexArrayView indices)
  : ConstituentItemIndexedSelectionViewBase(indices)
  , m_container_view(ecv)
  {
  }

  //! Construit une sélection contenant tous les éléments de \view
  explicit ConstituentItemIndexedSelectionView(ItemVecView view)
  requires(!IsSpanContainer())
  : ConstituentItemIndexedSelectionViewBase(view.component(), TraitsType::size(view))
  , m_container_view(view)
  {
  }

  //! Constructeur à partir d'une vue de ConstituentCell, de MatCell ou EnvCell
  explicit ConstituentItemIndexedSelectionView(IMeshComponent* constituent, SmallSpan<const ValueType> ecv)
  requires(IsSpanContainer())
  : ConstituentItemIndexedSelectionViewBase(constituent, TraitsType::size(ecv))
  , m_container_view(ecv)
  {
  }

  // nombre total de mailles du milieu
  ARCCORE_HOST_DEVICE Int32 sourceSize() const { return TraitsType::size(m_container_view); }

  // vue sur le vecteur de EnvCell d'origine (toutes les mailles du milieu)
  ItemVecView sourceView() const { return m_container_view; }

  // Liste des indices de la sélection.
  IndexArrayView selectionView() const
  {
    return m_selection_view.constSmallView();
  }

  ARCCORE_HOST_DEVICE ValueType operator[](Int32 i) const
  {
    return item(i);
  }

  ARCCORE_HOST_DEVICE ValueType item(Int32 i) const
  {
    ARCANE_CHECK_AT(i, size());
    return TraitsType::item(m_container_view, m_selection_view[i]);
  }

 private:

  //! Vue sur les éléments d'origine
  ItemVecView m_container_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur les éléments d'un ConstituentItemIndexedSelectionView.
 */
template <typename ContainerView_>
class ConstituentItemIndexedSelectionEnumerator
{
 public:

  using SelectionType = ConstituentItemIndexedSelectionView<ContainerView_>;
  using ThatClass = ConstituentItemIndexedSelectionEnumerator;

  using ValueType = SelectionType::ValueType;

  friend class EnumeratorTracer;
  friend class EnumeratorBuilder<ValueType>;

 private:

  explicit ConstituentItemIndexedSelectionEnumerator(const SelectionType& v)
  : m_size(v.size())
  , m_container_with_selection(v)
  {}

 public:

  static ThatClass create(SelectionType container)
  {
    return ThatClass(container);
  }

 public:

  void operator++() { ++m_index; }
  bool hasNext() const { return m_index < m_size; }

  ValueType operator*() const
  {
    return m_container_with_selection.item(m_index);
  }

  Int32 index() const { return m_index; }

 private:

  Int32 m_index = 0;
  Int32 m_size = 0;
  SelectionType m_container_with_selection;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Enumérateur sur une sélection d'un constituant
inline ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>
arcaneImplCreateConstituentEnumerator(ComponentCell, ConstituentItemIndexedSelectionView<ComponentCellVectorView> container)
{
  return ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>::create(container);
}
//! Enumérateur sur une sélection d'un constituant
inline ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>
arcaneImplCreateConstituentEnumerator(ComponentCell, ConstituentItemIndexedSelectionView<EnvCellVectorView> container)
{
  ConstituentItemIndexedSelectionView<ComponentCellVectorView> c2(container.sourceView(), container.selectionView());
  return ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>::create(c2);
}
//! Enumérateur sur une sélection d'un milieu
inline ConstituentItemIndexedSelectionEnumerator<EnvCellVectorView>
arcaneImplCreateConstituentEnumerator(EnvCell, ConstituentItemIndexedSelectionView<EnvCellVectorView> container)
{
  return ConstituentItemIndexedSelectionEnumerator<EnvCellVectorView>::create(container);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
