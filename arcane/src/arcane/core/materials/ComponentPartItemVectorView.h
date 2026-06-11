// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentPartItemVectorView.h                               (C) 2000-2024 */
/*                                                                           */
/* View over a vector of component entity parts.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTPARTITEMVECTORVIEW_H
#define ARCANE_CORE_MATERIALS_COMPONENTPARTITEMVECTORVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/ComponentItemInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over pure or partial entities of a component.
 */
class ARCANE_CORE_EXPORT ComponentPartItemVectorView
{
  friend class MeshComponentPartData;
  friend class ComponentPartCellEnumerator;

 protected:

  /*!
   * \brief Constructs a view over a part of the component entities \a component.
   *
   * This constructor is generally not called directly. To construct
   * such a view, it is preferable to use the methods
   * IMeshComponent::pureItems(), IMeshComponent::impureItems() or
   * IMeshComponent::partItems().
   */
  ComponentPartItemVectorView(IMeshComponent* component, Int32 component_part_index,
                              Int32ConstArrayView value_indexes,
                              Int32ConstArrayView item_indexes,
                              const ConstituentItemLocalIdListView& constituent_list_view,
                              eMatPart part)
  : m_component(component)
  , m_component_part_index(component_part_index)
  , m_value_indexes(value_indexes)
  , m_item_indexes(item_indexes)
  , m_constituent_list_view(constituent_list_view)
  , m_part(part)
  {
  }

 public:

  //! Constructs an uninitialized view
  ComponentPartItemVectorView() = default;

 public:

  //! Number of entities in the view
  Integer nbItem() const { return m_value_indexes.size(); }

  //! Associated component
  IMeshComponent* component() const { return m_component; }

  // Index of the part of this component (equivalent to MatVarIndex::arrayIndex()).
  Int32 componentPartIndex() const { return m_component_part_index; }

  //! List of valueIndex() of the part
  Int32ConstArrayView valueIndexes() const { return m_value_indexes; }

  //! List of indices into \a itemsInternal() of the entities.
  Int32ConstArrayView itemIndexes() const { return m_item_indexes; }

  //! Part of the component.
  eMatPart part() const { return m_part; }

 protected:

  //! Internal list of constituent items
  const ConstituentItemLocalIdListView& constituentItemListView() const { return m_constituent_list_view; }

 private:

  //! Constituent manager
  IMeshComponent* m_component = nullptr;

  //! Constituent index for accessing partial values.
  Int32 m_component_part_index = -1;

  //! List of valueIndex() of the part
  Int32ConstArrayView m_value_indexes;

  //! List of indices into \a m_items_internal for each material cell.
  Int32ConstArrayView m_item_indexes;

  //! List of ComponentItemInternal* for this constituent.
  ConstituentItemLocalIdListView m_constituent_list_view;

  //! Part of the constituent
  eMatPart m_part = eMatPart::Pure;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over the pure part of a component.
 */
class ARCANE_CORE_EXPORT ComponentPurePartItemVectorView
: public ComponentPartItemVectorView
{
  friend class MatPurePartItemVectorView;
  friend class EnvPurePartItemVectorView;
  friend class MeshComponentPartData;

 private:

  //! Constructs a view over a part of the component entities \a component.
  ComponentPurePartItemVectorView(IMeshComponent* component,
                                  Int32ConstArrayView value_indexes,
                                  Int32ConstArrayView item_indexes,
                                  const ConstituentItemLocalIdListView& constituent_list_view)
  : ComponentPartItemVectorView(component, 0, value_indexes, item_indexes, constituent_list_view, eMatPart::Pure)
  {
  }

 public:

  //! Constructs an uninitialized view
  ComponentPurePartItemVectorView() = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over the impure part of a component.
 */
class ARCANE_CORE_EXPORT ComponentImpurePartItemVectorView
: public ComponentPartItemVectorView
{
  friend class MatImpurePartItemVectorView;
  friend class EnvImpurePartItemVectorView;
  friend class MeshComponentPartData;

 private:

  //! Constructs a view over a part of the component entities \a component.
  ComponentImpurePartItemVectorView(IMeshComponent* component,
                                    Int32 component_part_index,
                                    Int32ConstArrayView value_indexes,
                                    Int32ConstArrayView item_indexes,
                                    const ConstituentItemLocalIdListView& constituent_list_view)
  : ComponentPartItemVectorView(component, component_part_index, value_indexes,
                                item_indexes, constituent_list_view, eMatPart::Impure)
  {
  }

 public:

  //! Constructs an uninitialized view
  ComponentImpurePartItemVectorView() = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over pure or partial entities of a material.
 */
class ARCANE_CORE_EXPORT MatPartItemVectorView
: public ComponentPartItemVectorView
{
 public:

  //! Constructs a view for the material \a material.
  MatPartItemVectorView(IMeshMaterial* material, const ComponentPartItemVectorView& view);

  //! Constructs an uninitialized view
  MatPartItemVectorView() = default;

 public:

  //! Associated material
  IMeshMaterial* material() const { return m_material; }

 private:

  IMeshMaterial* m_material = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over the pure part of the entities of a material.
 */
class ARCANE_CORE_EXPORT MatPurePartItemVectorView
: public MatPartItemVectorView
{
 public:

  //! Constructs a view for the material \a material.
  MatPurePartItemVectorView(IMeshMaterial* material, const ComponentPurePartItemVectorView& view)
  : MatPartItemVectorView(material, view)
  {}

  //! Constructs an uninitialized view
  MatPurePartItemVectorView()
  : MatPartItemVectorView()
  {}

 public:

  operator ComponentPurePartItemVectorView() const
  {
    return { component(), valueIndexes(),
             itemIndexes(), constituentItemListView() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over the impure part of the entities of a material.
 */
class ARCANE_CORE_EXPORT MatImpurePartItemVectorView
: public MatPartItemVectorView
{
 public:

  //! Constructs a view for the material \a material.
  MatImpurePartItemVectorView(IMeshMaterial* material, const ComponentImpurePartItemVectorView& view)
  : MatPartItemVectorView(material, view)
  {}

  //! Constructs an uninitialized view
  MatImpurePartItemVectorView()
  : MatPartItemVectorView()
  {}

 public:

  operator ComponentImpurePartItemVectorView() const
  {
    return { component(), componentPartIndex(),
             valueIndexes(),
             itemIndexes(), constituentItemListView() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over pure or partial entities of an environment.
 */
class ARCANE_CORE_EXPORT EnvPartItemVectorView
: public ComponentPartItemVectorView
{
 public:

  //! Constructs a view for the environment \a env.
  EnvPartItemVectorView(IMeshEnvironment* env, const ComponentPartItemVectorView& view);

  //! Constructs an uninitialized view
  EnvPartItemVectorView() = default;

 public:

  //! Associated environment
  IMeshEnvironment* environment() const { return m_environment; }

 private:

  IMeshEnvironment* m_environment = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over the pure part of the entities of an environment.
 */
class ARCANE_CORE_EXPORT EnvPurePartItemVectorView
: public EnvPartItemVectorView
{
 public:

  //! Constructs a view for the environment \a env.
  EnvPurePartItemVectorView(IMeshEnvironment* env, const ComponentPurePartItemVectorView& view)
  : EnvPartItemVectorView(env, view)
  {}

  //! Constructs an uninitialized view
  EnvPurePartItemVectorView()
  : EnvPartItemVectorView()
  {}

 public:

  operator ComponentPurePartItemVectorView() const
  {
    return { component(), valueIndexes(),
             itemIndexes(), constituentItemListView() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief View over the impure part of the entities of an environment.
 */
class ARCANE_CORE_EXPORT EnvImpurePartItemVectorView
: public EnvPartItemVectorView
{
 public:

  //! Constructs a view for the environment \a env.
  EnvImpurePartItemVectorView(IMeshEnvironment* env, const ComponentImpurePartItemVectorView& view)
  : EnvPartItemVectorView(env, view)
  {}

  //! Constructs an uninitialized view
  EnvImpurePartItemVectorView()
  : EnvPartItemVectorView()
  {}

 public:

  operator ComponentImpurePartItemVectorView() const
  {
    return { component(), componentPartIndex(),
             valueIndexes(),
             itemIndexes(), constituentItemListView() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
