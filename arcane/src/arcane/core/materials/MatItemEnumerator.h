// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemEnumerator.h                                         (C) 2000-2026 */
/*                                                                           */
/* Enumerators for material cells.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATITEMENUMERATOR_H
#define ARCANE_CORE_MATERIALS_MATITEMENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/IEnumeratorTracer.h"

#include "arcane/core/EnumeratorTraceWrapper.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Always enables tracing in Arcane parts concerning materials.
#ifdef ARCANE_COMPONENT_arcane_materials
#ifndef ARCANE_TRACE_ENUMERATOR
#define ARCANE_TRACE_ENUMERATOR
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MatCellVector;
class EnvCellVector;
class ComponentItemVector;
class ComponentItemVectorView;
class ComponentPartItemVectorView;
class MatItemVectorView;
class EnvItemVectorView;

// Les 4 classes suivantes servent uniquement pour spécialiser
// ComponentItemEnumeratorTraitsT
class MatPartCell
{};
class EnvPartCell
{};
class ComponentPartCell
{};
class ComponentPartSimdCell
{};

template <typename T> class ComponentItemEnumeratorTraitsT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief View over a list of cells with environment information.
 *
 * Like any view, this object is only valid as long as the container
 * associated (generally a CellGroup) is not modified.
 */
class ARCANE_CORE_EXPORT AllEnvCellVectorView
{
  friend class MeshMaterialMng;

 protected:

  AllEnvCellVectorView(Int32ConstArrayView local_ids,
                       ComponentItemSharedInfo* shared_info)
  : m_local_ids(local_ids)
  , m_shared_info(shared_info)
  {
  }

 public:

  //! Number of elements.
  constexpr ARCCORE_HOST_DEVICE Integer size() const { return m_local_ids.size(); }

  // i-th cell of the vector
  ARCCORE_HOST_DEVICE AllEnvCell operator[](Integer index) const
  {
    return AllEnvCell(m_shared_info->_item(ConstituentItemIndex(m_local_ids[index])));
  }

  // localId() of the i-th cell of the vector
  ARCCORE_HOST_DEVICE Int32 localId(Integer index) const { return m_local_ids[index]; }

 private:

  Int32ConstArrayView m_local_ids;
  ComponentItemSharedInfo* m_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a part of the cells of a component (material or environment)
 */
class ARCANE_CORE_EXPORT ComponentCellEnumerator
{
  friend class EnumeratorTracer;

 protected:

  explicit ComponentCellEnumerator(const ComponentItemVectorView& v);

 public:

  static ComponentCellEnumerator create(IMeshComponent* component);
  static ComponentCellEnumerator create(const ComponentItemVector& v);
  static ComponentCellEnumerator create(ComponentItemVectorView v);

 public:

  void operator++()
  {
    ++m_index;
#ifdef ARCANE_CHECK
    if (m_index < m_size)
      _check();
#endif
  }
  bool hasNext() const { return m_index < m_size; }

  ComponentCell operator*() const
  {
    return ComponentCell(m_constituent_list_view._constituenItemBase(m_index));
  }

  Integer index() const { return m_index; }
  MatVarIndex _varIndex() const { return m_matvar_indexes[m_index]; }

  operator ComponentItemLocalId() const
  {
    return ComponentItemLocalId(m_matvar_indexes[m_index]);
  }

 protected:

  Int32 _varArrayIndex() const { return m_matvar_indexes[m_index].arrayIndex(); }
  Int32 _varValueIndex() const { return m_matvar_indexes[m_index].valueIndex(); }

 protected:

  void _check() const
  {
    MatVarIndex mvi = m_constituent_list_view._matVarIndex(m_index);
    Int32 i_var_array_index = mvi.arrayIndex();
    Int32 mv_array_index = _varArrayIndex();
    if (i_var_array_index != mv_array_index)
      ARCANE_FATAL("Bad 'var_array_index' in ComponentCell matvar='{0}' registered='{1}' index={2}",
                   mvi, m_matvar_indexes[m_index], m_index);
    Int32 i_var_value_index = mvi.valueIndex();
    Int32 mv_value_index = _varValueIndex();
    if (i_var_value_index != mv_value_index)
      ARCANE_FATAL("Bad 'var_value_index' for ComponentCell matvar='{0}' registered='{1}' index={2}",
                   mvi, m_matvar_indexes[m_index], m_index);
  }

 protected:

  Int32 m_index;
  Int32 m_size;
  ConstituentItemLocalIdListView m_constituent_list_view;
  ConstArrayView<MatVarIndex> m_matvar_indexes;
  IMeshComponent* m_component;

 protected:

  matimpl::ConstituentItemBase _currentConstituentItemBase() const
  {
    return m_constituent_list_view._constituenItemBase(m_index);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a part of the cells of a single material.
 */
class ARCANE_CORE_EXPORT MatCellEnumerator
: public ComponentCellEnumerator
{
 protected:

  explicit MatCellEnumerator(const ComponentItemVectorView& v)
  : ComponentCellEnumerator(v)
  {
  }

 public:

  static MatCellEnumerator create(IMeshMaterial* mat);
  static MatCellEnumerator create(const MatCellVector& miv);
  static MatCellEnumerator create(MatItemVectorView v);

 public:

  MatCell operator*() const
  {
#ifdef ARCANE_CHECK
    _check();
#endif
    return MatCell(_currentConstituentItemBase());
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over the cells of an environment
 */
class ARCANE_CORE_EXPORT EnvCellEnumerator
: public ComponentCellEnumerator
{
 protected:

  explicit EnvCellEnumerator(const ComponentItemVectorView& v)
  : ComponentCellEnumerator(v)
  {
  }

 public:

  static EnvCellEnumerator create(IMeshEnvironment* mat);
  static EnvCellEnumerator create(const EnvCellVector& miv);
  static EnvCellEnumerator create(EnvItemVectorView v);

 public:

  EnvCell operator*() const
  {
#ifdef ARCANE_CHECK
    _check();
#endif
    return EnvCell(_currentConstituentItemBase());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over a sub-part (pure or partial) of a
 * subset of the cells of a component (material or environment)
 */
class ARCANE_CORE_EXPORT ComponentPartCellEnumerator
{
 protected:

  ComponentPartCellEnumerator(const ComponentPartItemVectorView& view, Integer base_index);

 public:

  static ComponentPartCellEnumerator create(ComponentPartItemVectorView v);
  static ComponentPartCellEnumerator create(IMeshComponent* c, eMatPart part);

 public:

  void operator++()
  {
    ++m_index;
  }
  bool hasNext() const { return m_index < m_size; }

  MatVarIndex _varIndex() const { return MatVarIndex(m_var_idx, m_value_indexes[m_index]); }

  operator ComponentItemLocalId() const
  {
    return ComponentItemLocalId(_varIndex());
  }

  ComponentCell operator*() const
  {
    return ComponentCell(m_constituent_list_view._constituenItemBase(m_item_indexes[m_index]));
  }

 protected:

  Integer m_index;
  Integer m_size;
  Integer m_var_idx;
  Integer m_base_index;
  Int32ConstArrayView m_value_indexes;
  Int32ConstArrayView m_item_indexes;
  ConstituentItemLocalIdListView m_constituent_list_view;
  IMeshComponent* m_component;

 protected:

  matimpl::ConstituentItemBase _currentConstituentItemBase() const
  {
    return m_constituent_list_view._constituenItemBase(m_item_indexes[m_index]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator for pure or impure entities of a material.
 */
class ARCANE_CORE_EXPORT MatPartCellEnumerator
: public ComponentPartCellEnumerator
{
 public:

  explicit MatPartCellEnumerator(const MatPartItemVectorView& v);

 public:

  static MatPartCellEnumerator create(MatPartItemVectorView v);
  static MatPartCellEnumerator create(IMeshMaterial* mat, eMatPart part);

 public:

  MatCell operator*() const
  {
    return MatCell(_currentConstituentItemBase());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over pure or impure entities of an environment.
 */
class ARCANE_CORE_EXPORT EnvPartCellEnumerator
: public ComponentPartCellEnumerator
{
 public:

  explicit EnvPartCellEnumerator(const EnvPartItemVectorView& v);

 public:

  static EnvPartCellEnumerator create(EnvPartItemVectorView v);
  static EnvPartCellEnumerator create(IMeshEnvironment* env, eMatPart part);

 public:

  EnvCell operator*() const
  {
    return EnvCell(_currentConstituentItemBase());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over the cells of an environment
 */
class ARCANE_CORE_EXPORT CellGenericEnumerator
{
 public:

  static EnvCellEnumerator create(IMeshEnvironment* env);
  static EnvCellEnumerator create(const EnvCellVector& ecv);
  static EnvCellEnumerator create(EnvItemVectorView v);

  static MatCellEnumerator create(IMeshMaterial* mat);
  static MatCellEnumerator create(const MatCellVector& miv);
  static MatCellEnumerator create(MatItemVectorView v);

  static CellEnumerator create(CellVectorView v);
  static CellEnumerator create(const CellGroup& v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over all environment cells
 */
class ARCANE_CORE_EXPORT AllEnvCellEnumerator
{
  friend class EnumeratorTracer;

 protected:

  explicit AllEnvCellEnumerator(AllEnvCellVectorView items)
  : m_index(0)
  , m_size(items.size())
  , m_items(items)
  {}

 public:

  static AllEnvCellEnumerator create(AllEnvCellVectorView items);
  static AllEnvCellEnumerator create(IMeshMaterialMng* mng, const CellGroup& group);
  static AllEnvCellEnumerator create(IMeshMaterialMng* mng, const CellVectorView& view);
  static AllEnvCellEnumerator create(IMeshBlock* block);

 public:

  void operator++() { ++m_index; }
  bool hasNext() const { return m_index < m_size; }
  AllEnvCell operator*() { return m_items[m_index]; }
  Integer index() const { return m_index; }

 public:

  Integer m_index;
  Integer m_size;
  AllEnvCellVectorView m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over components
 */
class ARCANE_CORE_EXPORT ComponentEnumerator
{
  friend class EnumeratorTracer;

 public:

  explicit ComponentEnumerator(ConstArrayView<IMeshComponent*> components);

 public:

  bool hasNext() const { return m_index < m_size; }
  void operator++() { ++m_index; }
  IMeshComponent* operator*() const { return m_components[m_index]; }

 private:

  ConstArrayView<IMeshComponent*> m_components;
  Integer m_index;
  Integer m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over materials
 */
class ARCANE_CORE_EXPORT MatEnumerator
{
 public:

  explicit MatEnumerator(IMeshMaterialMng* mng);
  explicit MatEnumerator(IMeshEnvironment* env);
  explicit MatEnumerator(ConstArrayView<IMeshMaterial*> mats);

 public:

  bool hasNext() const { return m_index < m_size; }
  void operator++() { ++m_index; }
  IMeshMaterial* operator*() const { return m_mats[m_index]; }

 private:

  ConstArrayView<IMeshMaterial*> m_mats;
  Integer m_index;
  Integer m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerator over environments
 */
class ARCANE_CORE_EXPORT EnvEnumerator
{
 public:

  EnvEnumerator(IMeshMaterialMng* mng);
  EnvEnumerator(IMeshBlock* block);
  EnvEnumerator(ConstArrayView<IMeshEnvironment*> envs);

 public:

  bool hasNext() const { return m_index < m_size; }
  void operator++() { ++m_index; }
  IMeshEnvironment* operator*() const { return m_envs[m_index]; }

 private:

  ConstArrayView<IMeshEnvironment*> m_envs;
  Integer m_index;
  Integer m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ComponentItemEnumeratorTraitsT<ComponentCell>
{
 public:

  using EnumeratorType = ComponentCellEnumerator;
};
template <>
class ComponentItemEnumeratorTraitsT<MatCell>
{
 public:

  using EnumeratorType = MatCellEnumerator;
};
template <>
class ComponentItemEnumeratorTraitsT<MatPartCell>
{
 public:

  using EnumeratorType = MatPartCellEnumerator;
};
template <>
class ComponentItemEnumeratorTraitsT<EnvPartCell>
{
 public:

  using EnumeratorType = EnvPartCellEnumerator;
};
template <>
class ComponentItemEnumeratorTraitsT<EnvCell>
{
 public:

  using EnumeratorType = EnvCellEnumerator;
};
template <>
class ComponentItemEnumeratorTraitsT<ComponentPartCell>
{
 public:

  using EnumeratorType = ComponentPartCellEnumerator;
};
template <>
class ComponentItemEnumeratorTraitsT<ComponentPartSimdCell>
{
 public:

  using EnumeratorType = ComponentPartSimdCellEnumerator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Enumerator over AllEnvCell of \a items
inline AllEnvCellEnumerator
arcaneImplCreateConstituentEnumerator(AllEnvCell, AllEnvCellVectorView items)
{
  return AllEnvCellEnumerator::create(items);
}
//! Enumerator over AllEnvCell of cells in \a group
inline AllEnvCellEnumerator
arcaneImplCreateConstituentEnumerator(AllEnvCell, IMeshMaterialMng* mng, const CellGroup& group)
{
  return AllEnvCellEnumerator::create(mng, group);
}
//! Enumerator over AllEnvCell of cells in \a view
inline AllEnvCellEnumerator
arcaneImplCreateConstituentEnumerator(AllEnvCell, IMeshMaterialMng* mng, const CellVectorView& view)
{
  return AllEnvCellEnumerator::create(mng, view);
}
//! Enumerator over AllEnvCell of the \a block
inline AllEnvCellEnumerator
arcaneImplCreateConstituentEnumerator(AllEnvCell, IMeshBlock* block)
{
  return AllEnvCellEnumerator::create(block);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Enumerator over ComponentCell of constituent \a component
inline ComponentCellEnumerator
arcaneImplCreateConstituentEnumerator(ComponentCell, IMeshComponent* component)
{
  return ComponentCellEnumerator::create(component);
}
//! Enumerator over ComponentCell of vector \a v
inline ComponentCellEnumerator
arcaneImplCreateConstituentEnumerator(ComponentCell, const ComponentItemVector& v)
{
  return ComponentCellEnumerator::create(v);
}
//! Enumerator over ComponentCell of view \a v
ARCANE_CORE_EXPORT ComponentCellEnumerator
arcaneImplCreateConstituentEnumerator(ComponentCell, ComponentItemVectorView v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Enumerator over MatCell of material \a component
inline MatCellEnumerator
arcaneImplCreateConstituentEnumerator(MatCell, IMeshMaterial* component)
{
  return MatCellEnumerator::create(component);
}
//! Enumerator over MatCell of vector \a v
inline MatCellEnumerator
arcaneImplCreateConstituentEnumerator(MatCell, const MatCellVector& v)
{
  return MatCellEnumerator::create(v);
}
//! Enumerator over MatCell of view \a v
ARCANE_CORE_EXPORT MatCellEnumerator
arcaneImplCreateConstituentEnumerator(MatCell, MatItemVectorView v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Enumerator over EnvCell of environment \a component
inline EnvCellEnumerator
arcaneImplCreateConstituentEnumerator(EnvCell, IMeshEnvironment* component)
{
  return EnvCellEnumerator::create(component);
}
//! Enumerator over EnvCell of vector \a v
inline EnvCellEnumerator
arcaneImplCreateConstituentEnumerator(EnvCell, const EnvCellVector& v)
{
  return EnvCellEnumerator::create(v);
}
//! Enumerator over EnvCell of view \a v
ARCANE_CORE_EXPORT EnvCellEnumerator
arcaneImplCreateConstituentEnumerator(EnvCell, EnvItemVectorView v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_CORE_EXPORT ComponentPartCellEnumerator
arcaneImplCreateConstituentEnumerator(ComponentPartCell, ComponentPartItemVectorView v);

inline ComponentPartCellEnumerator
arcaneImplCreateConstituentEnumerator(ComponentPartCell, IMeshComponent* c, eMatPart part)
{
  return ComponentPartCellEnumerator::create(c, part);
}
ARCANE_CORE_EXPORT MatPartCellEnumerator
arcaneImplCreateConstituentEnumerator(MatPartCell, MatPartItemVectorView v);

inline MatPartCellEnumerator
arcaneImplCreateConstituentEnumerator(MatPartCell, IMeshMaterial* c, eMatPart part)
{
  return MatPartCellEnumerator::create(c, part);
}
ARCANE_CORE_EXPORT EnvPartCellEnumerator
arcaneImplCreateConstituentEnumerator(EnvPartCell, EnvPartItemVectorView v);

inline EnvPartCellEnumerator
arcaneImplCreateConstituentEnumerator(EnvPartCell, IMeshEnvironment* c, eMatPart part)
{
  return EnvPartCellEnumerator::create(c, part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MImpl
{

  /*!
 * \brief Wrapper function to detect the enumerator type for constituents.
 */
  template <typename ConstituentItemType, typename ConstituentItemContainerType, typename... RemainingArgs> auto
  makeConstituentItemEnumeratorLoop(ConstituentItemType x,
                                    const ConstituentItemContainerType& container,
                                    const RemainingArgs&... remaining_args)
  {
    auto container_instance = arcaneImplCreateConstituentEnumerator(x, container, remaining_args...);
    return container_instance;
  }

} // namespace MImpl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_TRACE_ENUMERATOR)
#define A_TRACE_COMPONENT(_EnumeratorClassName) \
  ::Arcane::EnumeratorTraceWrapper<::Arcane::Materials::_EnumeratorClassName, ::Arcane::Materials::IEnumeratorTracer>
#define A_TRACE_COMPONENT_DIRECT_CLASS(_EnumeratorClassName) \
  ::Arcane::EnumeratorTraceWrapper<_EnumeratorClassName, ::Arcane::Materials::IEnumeratorTracer>
#else
#define A_TRACE_COMPONENT(_EnumeratorClassName) \
  ::Arcane::Materials::_EnumeratorClassName
#define A_TRACE_COMPONENT_DIRECT_CLASS(_EnumeratorClassName) \
  _EnumeratorClassName
#endif

#define A_ENUMERATE_COMPONENTCELL_OLD(_EnumeratorClassName, iname, ...) \
  for (A_TRACE_COMPONENT(_EnumeratorClassName) iname(Arcane::Materials::_EnumeratorClassName::create(__VA_ARGS__) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname)

#define A_ENUMERATE_COMPONENT(_EnumeratorClassName, iname, container) \
  for (A_TRACE_COMPONENT(_EnumeratorClassName) iname((::Arcane::Materials::_EnumeratorClassName)(container)A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname)

#define A_ENUMERATE_CELL_COMPONENTCELL(_EnumeratorClassName, iname, component_cell) \
  for (::Arcane::Materials::_EnumeratorClassName iname((::Arcane::Materials::_EnumeratorClassName)(component_cell)); iname.hasNext(); ++iname)

#define A_ENUMERATEBUILDER_HELPER(ClassName_, ...) \
  ::Arcane::Materials::EnumeratorBuilder<::Arcane::Materials::ClassName_>::create(__VA_ARGS__)

#define A_ENUMERATEBUILDER_HELPER2(ConstituentItemNameType, env_or_mat_container, ...) \
  ::Arcane::Materials::MImpl::makeConstituentItemEnumeratorLoop(ConstituentItemNameType(), env_or_mat_container __VA_OPT__(, __VA_ARGS__))

#define A_ENUMERATE_COMPONENTCELL(ClassName_, iname, env_or_mat_container, ...) \
  for (A_TRACE_COMPONENT_DIRECT_CLASS(decltype(A_ENUMERATEBUILDER_HELPER2(ClassName_, env_or_mat_container __VA_OPT__(, __VA_ARGS__)))) \
       iname(A_ENUMERATEBUILDER_HELPER2(ClassName_, env_or_mat_container __VA_OPT__(, __VA_ARGS__)) A_TRACE_ENUMERATOR_WHERE); \
       iname.hasNext(); ++iname)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic macro to iterate over entities of a material or an environment.
 *
 * \param ClassName_ name of the constituent class (ConstituentCell, MatCell, or EnvCell)
 * \param iname name of the variable containing the iterator
 * \param container container to iterate over
 * \param ... Additional arguments are passed to the iterator creation method.
 *
 * This macro automatically selects an enumerator based on the parameters \a ClassName_ and \a container. For it to be valid, there must exist an overload of the function
 * Arcane::Materials::arcaneImplCreateConstituentEnumerator()
 * taking these two arguments.
 */
#define ENUMERATE_COMPONENTITEM(ClassName_, iname, container, ...) \
  A_ENUMERATE_COMPONENTCELL(ClassName_, iname, container, __VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to iterate over all AllEnvCell cells of a group.
 *
 * There are two ways to use this macro. The first
 * is obsolete and uses the IMeshMaterialMng::view() method. The second
 * uses three arguments:
 * \param iname name of the iterator, of type \a AllEnvCellEnumerator.
 * \param matmng material manager of type \a IMeshMaterialMng.
 * \param igroup cell group, of type \a CellGroup.
 */
#define ENUMERATE_ALLENVCELL(iname, ...) \
  for (A_TRACE_COMPONENT(AllEnvCellEnumerator) iname(::Arcane::Materials::AllEnvCellEnumerator::create(__VA_ARGS__)); iname.hasNext(); ++iname)

/*!
 * \brief Macro to iterate over all MatCell cells of a material.
 *
 * \param iname name of the iterator, of type MatCellEnumerator.
 * \param mat material, of type IMeshMaterial*, MatCellVector, MatVectorView
 */
#define ENUMERATE_MATCELL(iname, mat) \
  A_ENUMERATE_COMPONENTCELL(Arcane::Materials::MatCell, iname, mat)

/*!
 * \brief Macro to iterate over all EnvCell cells of an environment.
 *
 * \param iname name of the iterator, of type EnvCellEnumerator.
 * \param env environment, of type IMeshEnvironment*, EnvCellVector,
 * EvnVectorView or EnvCellVectorSelectionView
 */
#define ENUMERATE_ENVCELL(iname, env) \
  A_ENUMERATE_COMPONENTCELL(Arcane::Materials::EnvCell, iname, env)

/*!
 * \brief Macro to iterate over all ComponentCell cells of a component.
 *
 * \param iname name of the iterator, of type EnvCellEnumerator.
 * \param component component, of type IMeshComponent*
 */
#define ENUMERATE_COMPONENTCELL(iname, component) \
  A_ENUMERATE_COMPONENTCELL(Arcane::Materials::ComponentCell, iname, component)

/*!
 * \brief Macro to iterate over a list of components
 *
 * \a icomponent name of the iterator.
 * \a container can be an object of the following type:
 * - ConstArrayView<IMeshComponent*> to iterate over a specific list of components
 */
#define ENUMERATE_COMPONENT(icomponent, container) \
  A_ENUMERATE_COMPONENT(ComponentEnumerator, icomponent, container)

/*!
 * \brief Macro to iterate over a list of materials
 *
 * \a imat name of the iterator.
 * \a container can be an object of the following type:
 * - IMeshMaterialMng* to iterate over all materials
 * - IMeshEnvironment* to iterate over all materials of an environment
 * - ConstArrayView<IMeshMaterial*> to iterate over a specific list of materials
 */
#define ENUMERATE_MAT(imat, container) \
  A_ENUMERATE_COMPONENT(MatEnumerator, imat, container)

/*!
 * \brief Macro to iterate over a list of environments.
 *
 * \a ienv is the name of the enumerator.
 * \a container can be an object of the following type:
 * - IMeshMaterialMng* to iterate over all environments
 * - IMeshBlock* to iterate over all environments in the block
 * - ConstArrayView<IMeshEnvironment*> to iterate over a specific list of environments.
 */
#define ENUMERATE_ENV(ienv, container) \
  A_ENUMERATE_COMPONENT(EnvEnumerator, ienv, container)

/*!
 * \brief Macro to iterate over all ComponentCell cells of a cell.
 *
 * \param iname name of the iterator, of type CellComponentCellEnumerator.
 * \param component_cell cell component, of type ComponentCell.
 */
#define ENUMERATE_CELL_COMPONENTCELL(iname, component_cell) \
  A_ENUMERATE_CELL_COMPONENTCELL(CellComponentCellEnumerator, iname, component_cell)

/*!
 * \brief Macro to iterate over all MatCell cells of a cell.
 *
 * \param iname name of the iterator, of type CellMatCellEnumerator.
 * \param env_cell cell environment, of type EnvCell.
 */
#define ENUMERATE_CELL_MATCELL(iname, env_cell) \
  A_ENUMERATE_CELL_COMPONENTCELL(CellMatCellEnumerator, iname, env_cell)

/*!
 * \brief Macro to iterate over all EnvCell cells of a cell.
 *
 * \param iname name of the iterator, of type CellEnvCellEnumerator.
 * \param all_env_cell cell with environment info, of type AllEnvCell.
 */
#define ENUMERATE_CELL_ENVCELL(iname, all_env_cell) \
  A_ENUMERATE_CELL_COMPONENTCELL(CellEnvCellEnumerator, iname, all_env_cell)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro for generically iterating over materials,
 * environments, or cells
 *
 * \param iname name of the iterator, of type CellEnvCellEnumerator.
 * \param mat_or_env_or_group an object that can be passed as an argument to
 * ENUMERATE_CELL, ENUMERATE_MATCELL or ENUMERATE_ENVCELL.
 */
#define ENUMERATE_GENERIC_CELL(iname, mat_or_env_or_group) \
  for (auto iname = ::Arcane::Materials::CellGenericEnumerator::create(mat_or_env_or_group); iname.hasNext(); ++iname)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
