// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVectorView.h                                   (C) 2000-2026 */
/*                                                                           */
/* View over a vector of constituent entities.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEMVECTORVIEW_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEMVECTORVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemGroup.h"

#include "arcane/core/materials/MatVarIndex.h"
#include "arcane/core/materials/IMeshComponent.h"
#include "arcane/core/materials/ComponentItemInternal.h"
#include "arcane/core/materials/MatItem.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
class MeshMaterialTesterModule;
class MaterialHeatTestModule;
} // namespace ArcaneTest
namespace Arcane::Accelerator::Impl
{
class ConstituentCommandContainerBase;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View over a vector of entities of a component.
 *
 * The constructors of this class are internal to %Arcane.
 */
class ARCANE_CORE_EXPORT ComponentItemVectorView
{
  friend class ComponentItemVector;
  friend class ConstituentItemVectorImpl;
  friend class MatItemVectorView;
  friend class EnvItemVectorView;
  friend class MatCellEnumerator;
  friend class EnvCellEnumerator;
  friend class ComponentCellEnumerator;
  friend Arcane::Accelerator::Impl::ConstituentCommandContainerBase;
  friend ArcaneTest::MeshMaterialTesterModule;
  friend ArcaneTest::MaterialHeatTestModule;
  template <typename ViewType, typename LambdaType>
  friend class LambdaMatItemRangeFunctorT;
  template <typename DataType> friend class
  MaterialVariableArrayTraits;

 public:

  using ValueType = ComponentCell;

 public:

  ComponentItemVectorView() = default;

 protected:

  //! Constructs a vector containing the entities of \a group for the component \a component
  ComponentItemVectorView(IMeshComponent* component,
                          ConstArrayView<MatVarIndex> mvi,
                          ConstituentItemLocalIdListView constituent_local_ids,
                          ConstArrayView<Int32> local_ids)
  : m_matvar_indexes_view(mvi)
  , m_constituent_list_view(constituent_local_ids)
  , m_items_local_id_view(local_ids)
  , m_component(component)
  {
  }

  //! Constructs an empty view for the component \a component
  explicit ComponentItemVectorView(IMeshComponent* component)
  : m_component(component)
  {
  }

  //! Constructs a view from another view.
  ComponentItemVectorView(IMeshComponent* component, ComponentItemVectorView rhs_view)
  : m_matvar_indexes_view(rhs_view.m_matvar_indexes_view)
  , m_constituent_list_view(rhs_view.m_constituent_list_view)
  , m_items_local_id_view(rhs_view.m_items_local_id_view)
  , m_component(component)
  {
  }

 public:

  //! Number of entities in the view
  Integer nbItem() const { return m_matvar_indexes_view.size(); }

  //! Associated component
  IMeshComponent* component() const { return m_component; }

  //! Returns the index-th ComponentCell of the view
  ARCCORE_HOST_DEVICE ComponentCell componentCell(Int32 index) const
  {
    return m_constituent_list_view._constituenItemBase(index);
  }

 private:

  // Array of MatVarIndex for this view.
  ConstArrayView<MatVarIndex> _matvarIndexes() const { return m_matvar_indexes_view; }

  //! Array of localId() of associated entities
  ConstArrayView<Int32> _internalLocalIds() const { return m_items_local_id_view; }

  ConstituentItemLocalIdListView _constituentItemListView() const { return m_constituent_list_view; }

  /*!
   * \internal
   * \brief Creates a sub-view of this view.
   * 
   * This method is internal to Arcane and should not be used.
   */
  ComponentItemVectorView _subView(Integer begin, Integer size);

  //! For tests verifies that \a rhs and the instance point to the same data
  bool _isSamePointerData(const ComponentItemVectorView& rhs) const
  {
    bool test1 = m_constituent_list_view._isSamePointerData(rhs.m_constituent_list_view);
    return test1 && (m_matvar_indexes_view.data() == rhs.m_matvar_indexes_view.data());
  }

 private:

  // NOTE: This class is wrapped directly in C#.
  // If the fields of this class are modified, the corresponding type
  // in the wrapper must be modified.
  ConstArrayView<MatVarIndex> m_matvar_indexes_view;
  ConstituentItemLocalIdListView m_constituent_list_view;
  ConstArrayView<Int32> m_items_local_id_view;
  IMeshComponent* m_component = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View over a vector of entities of a material.
 *
 * The constructors of this class are internal to %Arcane.
 */
class ARCANE_CORE_EXPORT MatItemVectorView
: public ComponentItemVectorView
{
  friend class MatCellVector;
  friend class MeshMaterial;
  template <typename ViewType, typename LambdaType>
  friend class LambdaMatItemRangeFunctorT;

 public:

  using ValueType = MatCell;

 public:

  MatItemVectorView() = default;

 private:

  MatItemVectorView(IMeshComponent* component,
                    ConstArrayView<MatVarIndex> mv_indexes,
                    ConstituentItemLocalIdListView constituent_local_ids,
                    ConstArrayView<Int32> local_ids)
  : ComponentItemVectorView(component, mv_indexes, constituent_local_ids, local_ids)
  {}

  MatItemVectorView(IMeshComponent* component, ComponentItemVectorView v)
  : ComponentItemVectorView(component, v)
  {}

 private:

  /*!
   * \internal
   * \brief Creates a sub-view of this view.
   * 
   * This method is internal to Arcane and should not be used.
   */
  MatItemVectorView _subView(Integer begin, Integer size);

 public:

  //! Associated material
  IMeshMaterial* material() const;

  //! Retrieves the index-th MatCell of the view
  ARCCORE_HOST_DEVICE MatCell matCell(Int32 index) const { return MatCell(componentCell(index)); }

  // Temporary: kept for compatibility
  ARCANE_DEPRECATED_240 MatItemVectorView subView(Integer begin, Integer size)
  {
    return _subView(begin, size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View over a vector of entities of an environment.
 *
 * The constructors of this class are internal to %Arcane.
 */
class ARCANE_CORE_EXPORT EnvItemVectorView
: public ComponentItemVectorView
{
  friend class EnvCellVector;
  friend class MeshEnvironment;
  template <typename ViewType, typename LambdaType>
  friend class LambdaMatItemRangeFunctorT;

 public:

  using ValueType = EnvCell;

 public:

  EnvItemVectorView() = default;

 private:

  EnvItemVectorView(IMeshComponent* component,
                    ConstArrayView<MatVarIndex> mv_indexes,
                    ConstituentItemLocalIdListView constituent_local_ids,
                    ConstArrayView<Int32> local_ids)
  : ComponentItemVectorView(component, mv_indexes, constituent_local_ids, local_ids)
  {}

  EnvItemVectorView(IMeshComponent* component, ComponentItemVectorView v)
  : ComponentItemVectorView(component, v)
  {}

 private:

  /*!
   * \internal
   * \brief Creates a sub-view of this view.
   * 
   * This method is internal to Arcane and should not be used.
   */
  EnvItemVectorView _subView(Integer begin, Integer size);

 public:

  //! Associated environment
  IMeshEnvironment* environment() const;

  //! Retrieves the index-th EnvCell of the view
  ARCCORE_HOST_DEVICE EnvCell envCell(Int32 index) const { return EnvCell(componentCell(index)); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
