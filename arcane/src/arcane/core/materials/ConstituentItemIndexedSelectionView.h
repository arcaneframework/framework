// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemIndexedSelectionView.h                       (C) 2000-2026 */
/*                                                                           */
/* View over a subset of a ConstituentItem container.                        */
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
 * \brief Characteristics for the container associated with
 * ConstituentItemIndexedSelectionView.
 *
 * This class must be specialized. It is specialized for containers
 * of type ComponentItemVectorView, MatCellVectorView, EnvCellVectorView
 * or SmallSpan<T>, where T is of type ComponentCell, MatCell or EnvCell.
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

//! Partial specialization for a SmallSpan<T>
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
 * \brief Base class for ConstituentItemIndexedSelectionView.
 */
class ARCANE_CORE_EXPORT ConstituentItemIndexedSelectionViewBase
{
 public:

  using IndexArrayView = const SmallSpan<const Int32>;

 protected:

  explicit ConstituentItemIndexedSelectionViewBase(SmallSpan<const Int32> indices);
  explicit ConstituentItemIndexedSelectionViewBase(IMeshComponent* constituent, Int32 selection_size);

 public:

  //! number of selected EnvCells
  ARCCORE_HOST_DEVICE Int32 size() const { return m_selection_view.size(); }

 protected:

  /*!
   * \brief Selection.
   *
   * If this field is omitted during construction, the default will be a 'full'
   * selection (i.e., all original elements, in the same order)
   */
  SmallSpan<const Int32> m_selection_view = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View over a subset of a ConstituentItem container.
 *
 * The container is the template argument \a ContainerView_. It can
 * be a ComponentItemVectorView, MatCellVectorView, EnvCellVectorView
 * or just a SmallSpan of a ConstituentItem.
 * Entity selection is done by an array of indices.
 * If this array is not provided, the selection is on all entities.
 *
 * Like any view, instances of this class are invalidated if the
 * constituents change (addition or deletion)
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

  //! Constructor from a view of ConstituentCell, MatCell or EnvCell
  explicit ConstituentItemIndexedSelectionView(IMeshComponent* constituent, SmallSpan<const ValueType> ecv)
  requires(IsSpanContainer())
  : ConstituentItemIndexedSelectionViewBase(constituent, TraitsType::size(ecv))
  , m_container_view(ecv)
  {
  }

 protected:

  //! Constructs a selection containing all elements of \view (which must derive from ComponentCellVectorView)
  explicit ConstituentItemIndexedSelectionView(ItemVecView view)
  : ConstituentItemIndexedSelectionViewBase(view.component(), TraitsType::size(view))
  , m_container_view(view)
  {
  }

 public:

  // total number of medium meshes
  ARCCORE_HOST_DEVICE Int32 sourceSize() const { return TraitsType::size(m_container_view); }

  // view over the original EnvCell vector (all medium meshes)
  ItemVecView sourceView() const { return m_container_view; }

  // List of selection indices.
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

  //! View over the original elements
  ItemVecView m_container_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over the elements of a ConstituentItemIndexedSelectionView.
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

/*!
 * \brief Selection over a ComponentCellVectorView.
 */
class ComponentCellVectorSelectionView
: public ConstituentItemIndexedSelectionView<ComponentCellVectorView>
{
  using BaseClass = ConstituentItemIndexedSelectionView<ComponentCellVectorView>;

 public:

  ComponentCellVectorSelectionView(ComponentCellVectorView vector_view, SmallSpan<const Int32> indices)
  : BaseClass(vector_view, indices)
  {
  }

  //! Constructs a selection containing all elements of \view
  explicit ComponentCellVectorSelectionView(ComponentCellVectorView vector_view)
  : BaseClass(vector_view)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Selection over an EnvCellVectorView.
 */
class EnvCellVectorSelectionView
: public ConstituentItemIndexedSelectionView<EnvCellVectorView>
{
  using BaseClass = ConstituentItemIndexedSelectionView<EnvCellVectorView>;

 public:

  EnvCellVectorSelectionView(EnvCellVectorView vector_view, SmallSpan<const Int32> indices)
  : BaseClass(vector_view, indices)
  {
  }

  //! Constructs a selection containing all elements of \view
  explicit EnvCellVectorSelectionView(EnvCellVectorView vector_view)
  : BaseClass(vector_view)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Selection over a MatCellVectorView.
 */
class MatCellVectorSelectionView
: public ConstituentItemIndexedSelectionView<MatCellVectorView>
{
  using BaseClass = ConstituentItemIndexedSelectionView<MatCellVectorView>;

 public:

  MatCellVectorSelectionView(MatCellVectorView vector_view, SmallSpan<const Int32> indices)
  : BaseClass(vector_view, indices)
  {
  }

  //! Constructs a selection containing all elements of \view
  explicit MatCellVectorSelectionView(MatCellVectorView vector_view)
  : BaseClass(vector_view)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Enumerator over a constituent selection
inline ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>
arcaneImplCreateConstituentEnumerator(ComponentCell, ComponentCellVectorSelectionView container)
{
  return ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>::create(container);
}

//! Enumerator over a constituent selection
inline ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>
arcaneImplCreateConstituentEnumerator(ComponentCell, EnvCellVectorSelectionView container)
{
  ConstituentItemIndexedSelectionView<ComponentCellVectorView> c2(container.sourceView(), container.selectionView());
  return ConstituentItemIndexedSelectionEnumerator<ComponentCellVectorView>::create(c2);
}

//! Enumerator over a medium selection
inline ConstituentItemIndexedSelectionEnumerator<EnvCellVectorView>
arcaneImplCreateConstituentEnumerator(EnvCell, EnvCellVectorSelectionView container)
{
  return ConstituentItemIndexedSelectionEnumerator<EnvCellVectorView>::create(container);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
