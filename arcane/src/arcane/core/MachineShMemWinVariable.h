// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariable.h                                   (C) 2000-2026 */
/*                                                                           */
/* Classes allowing the use of the MachineShMemWinVariable object pointed    */
/*  to by the shared memory variable memory area.                            */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_MACHINESHMEMWINVARIABLE_H
#define ARCANE_CORE_MACHINESHMEMWINVARIABLE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/NumArray.h"

#include "arcane/core/MeshMDVariableRef.h"

#include "arccore/base/FixedArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MachineShMemWinVariableBase;
class MachineShMemWinVariable2DBase;
class MachineShMemWinVariableMDBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared elements of the variable
 * in shared memory.
 *
 * To access all properties, it is necessary to use one of the child classes:
 * - \a MachineShMemWinVariableArrayT for array variables without
 *   support,
 * - \a MachineShMemWinVariableItemT for mesh variables.
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariableCommon
{

 protected:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableCommon(IVariable* var);

 public:

  virtual ~MachineShMemWinVariableCommon();

 public:

  /*!
   * \brief Method allowing retrieval of ranks that possess a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method allowing waiting until all processes/threads
   * on the node call this method to continue execution.
   */
  void barrier() const;

 protected:

  Ref<MachineShMemWinVariableBase> m_base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared elements of the variable
 * in shared memory.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * This class works for array variables without support.
 *
 * If the size of the variable changes while an object of this type is used,
 * it is necessary to call the \a updateVariable() method.
 */
template <class DataType>
class MachineShMemWinVariableArrayT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinVariableArrayT(VariableRefArrayT<DataType> var);
  ARCANE_CORE_EXPORT ~MachineShMemWinVariableArrayT() override;

 public:

  /*!
   * \brief Method allowing retrieval of a view on the array of another
   * subdomain on the node.
   *
   * Equivalent to "var.asArray()" but for another subdomain.
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  ARCANE_CORE_EXPORT Span<DataType> view(Int32 rank) const;

  /*!
   * \brief Method allowing updating this object after a
   * resizing of the variable.
   *
   * Collective call.
   */
  ARCANE_CORE_EXPORT void updateVariable();

 private:

  VariableRefArrayT<DataType> m_vart;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared elements of the variable
 * in shared memory.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * This class works for mesh scalar variables.
 *
 * If the mesh changes while an object of this type is used, it is
 * necessary to call the \a updateVariable() method.
 */
template <class ItemType, class DataType>
class MachineShMemWinMeshVariableScalarT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinMeshVariableScalarT(MeshVariableScalarRefT<ItemType, DataType> var);

  ARCANE_CORE_EXPORT ~MachineShMemWinMeshVariableScalarT() override;

 public:

  /*!
   * \brief Method allowing retrieval of a view on the variable of another
   * subdomain on the node.
   *
   * Equivalent to "var.asArray()" but for another subdomain.
   *
   * \warning Attention: To access the elements of the view, it is
   *          necessary to use the local_ids of the other subdomain!
   *          Do not use the local_ids of our subdomain!
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  ARCANE_CORE_EXPORT Span<DataType> view(Int32 rank) const;

  /*!
   * \brief Method allowing retrieval of an element of the variable from another
   * subdomain.
   *
   * \warning Attention: The local_id corresponds to the local_id of the subdomain
   *          \a rank! Absolutely do not use a local_id from our
   *          subdomain to access the elements of the view!
   *
   * \note If multiple iterations are necessary for the same rank, it is
   *       preferable to retrieve a view via \a segmentView(Int32 rank).
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain of the targeted variable.
   * \param notlocal_id The local_id of the subdomain \a rank.
   * \return The item element.
   */
  ARCANE_CORE_EXPORT DataType operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Method allowing updating this object after a change
   * in the mesh.
   *
   * Collective call.
   */
  ARCANE_CORE_EXPORT void updateVariable();

 private:

  MeshVariableScalarRefT<ItemType, DataType> m_vart;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared elements of the variable
 * in shared memory.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * This class works for 2D array variables without support.
 *
 * If the size of the variable changes while an object of this type is used,
 * it is necessary to call the \a updateVariable() method.
 */
template <class DataType>
class MachineShMemWinVariableArray2T
{
 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinVariableArray2T(VariableRefArray2T<DataType> var);

  ARCANE_CORE_EXPORT ~MachineShMemWinVariableArray2T();

 public:

  /*!
   * \brief Method allowing retrieval of ranks that possess a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  ARCANE_CORE_EXPORT ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method allowing waiting until all processes/threads
   * on the node call this method to continue execution.
   */
  ARCANE_CORE_EXPORT void barrier() const;

 public:

  /*!
   * \brief Method allowing retrieval of a view on the array of another
   * subdomain on the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain.
   * \return A 2D view.
   */
  ARCANE_CORE_EXPORT Span2<DataType> view(Int32 rank) const;

  /*!
   * \brief Method allowing updating this object after a
   * resizing of the variable.
   *
   * Collective call.
   */
  ARCANE_CORE_EXPORT void updateVariable();

 private:

  Ref<MachineShMemWinVariable2DBase> m_base;
  VariableRefArray2T<DataType> m_vart;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared elements of the variable
 * in shared memory.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * This class works for mesh array variables.
 *
 * If the mesh and/or the variable size changes when an object of this
 * type is used, it is necessary to call the \a updateVariable() method.
 */
template <class ItemType, class DataType>
class MachineShMemWinMeshVariableArrayT
{

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinMeshVariableArrayT(MeshVariableArrayRefT<ItemType, DataType> var);

  ARCANE_CORE_EXPORT ~MachineShMemWinMeshVariableArrayT();

 public:

  /*!
   * \brief Method to get the ranks that possess a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  ARCANE_CORE_EXPORT ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method to wait until all processes/threads
   * on the node call this method to continue execution.
   */
  ARCANE_CORE_EXPORT void barrier() const;

 public:

  /*!
   * \brief Method to get a view of the variable from another
   * subdomain on the node.
   *
   * Equivalent to "var.asArray()" but from another subdomain.
   * The first index corresponds to the local_id, the second index is the
   * position of the element in the item array.
   *
   * \warning Attention: to access the elements of the view, it is
   *          necessary to use the local_ids of the other subdomain!
   *          Do not use the local_ids of our subdomain!
   *
   * Non-collective call.
   *
   * \param rank The subdomain rank.
   * \return A 2D view.
   */
  ARCANE_CORE_EXPORT Span2<DataType> view(Int32 rank) const;

  /*!
   * \brief Method to get the array of an item from another
   * subdomain.
   *
   * \warning Attention: the local_id corresponds to the local_id of the subdomain
   *          \a rank! Absolutely do not use a local_id from our
   *          subdomain to access the elements of the view!
   *
   * \note If multiple iterations are necessary for the same rank, it is
   *       preferable to retrieve a view via \a segmentView(Int32 rank).
   *
   * Non-collective call.
   *
   * \param rank The rank of the targeted variable's subdomain.
   * \param notlocal_id The local_id of the subdomain \a rank.
   * \return The item array.
   */
  ARCANE_CORE_EXPORT Span<DataType> operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Method to update this object after a change
   * in the mesh and/or after a resizing of the variable.
   *
   * Collective call.
   */
  ARCANE_CORE_EXPORT void updateVariable();

 private:

  Ref<MachineShMemWinVariableMDBase> m_base;
  MeshVariableArrayRefT<ItemType, DataType> m_vart;
  Int32 m_nb_elem_dim2{};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to shared elements of the variable
 * in shared memory.
 *
 * This class cannot be used directly. It is necessary
 * to use one of the following classes:
 * - \a MachineShMemWinMeshMDVariableT for scalar mesh variables
 *   with a maximum dimension of 3,
 * - \a MachineShMemWinMeshVectorMDVariableT for vector mesh variables
 *   with a maximum dimension of 2,
 * - \a MachineShMemWinMeshMatrixMDVariableT for matrix mesh variables
 *   with a maximum dimension of 1.
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMDVariableT
{

 protected:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinMDVariableT(MeshVariableArrayRefT<ItemType, DataType> var);

 public:

  ARCANE_CORE_EXPORT virtual ~MachineShMemWinMDVariableT();

 public:

  /*!
   * \brief Method to get the ranks that possess a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  ARCANE_CORE_EXPORT ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method to wait until all processes/threads
   * on the node call this method to continue execution.
   */
  ARCANE_CORE_EXPORT void barrier() const;

 public:

  /*!
   * \brief Method to get a view of the variable from another
   * subdomain on the node.
   *
   * The first index corresponds to the local_id, the other indices are the
   * position of the element in the item array.
   *
   * \warning Attention: to access the elements of the view, it is
   *          necessary to use the local_ids of the other subdomain!
   *          Do not use the local_ids of our subdomain!
   *
   * Non-collective call.
   *
   * \param rank The subdomain rank.
   * \return A view.
   */
  ARCANE_CORE_EXPORT MDSpan<DataType, typename MDDimType<Extents::rank() + 1>::DimType> view(Int32 rank) const;

  /*!
   * \brief Method to get the multi-dimensional array of an
   * item from another subdomain.
   *
   * \warning Attention: the local_id corresponds to the local_id of the subdomain
   *          \a rank! Absolutely do not use a local_id from our
   *          subdomain to access the elements of the view!
   *
   * \note If multiple iterations are necessary for the same rank, it is
   *       preferable to retrieve a view via \a view(Int32 rank).
   *
   * Non-collective call.
   *
   * \param rank The rank of the targeted variable's subdomain.
   * \param notlocal_id The local_id of the subdomain \a rank.
   * \return The MD array of the item.
   */
  ARCANE_CORE_EXPORT MDSpan<DataType, Extents> operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Method to update this object after a change
   * in the mesh and/or after a resizing of the variable.
   *
   * Collective call.
   */
  ARCANE_CORE_EXPORT void updateVariable();

 private:

  Ref<MachineShMemWinVariableMDBase> m_base;
  MeshVariableArrayRefT<ItemType, DataType> m_vart;
  Int32 m_nb_elem_dim2{};
  std::array<Int32, Extents::rank()> m_shape_dim2{};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to shared elements of the variable
 * in shared memory.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * This class works for scalar mesh variables
 * with a maximum dimension of 3.
 *
 * If the mesh and/or the variable size changes when an object of this
 * type is used, it is necessary to call the \a updateVariable() method.
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMeshMDVariableT
: public MachineShMemWinMDVariableT<ItemType, DataType, Extents>
{

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  explicit MachineShMemWinMeshMDVariableT(MeshMDVariableRefT<ItemType, DataType, Extents> var)
  : MachineShMemWinMDVariableT<ItemType, DataType, Extents>(var.underlyingVariable())
  {}

  ~MachineShMemWinMeshMDVariableT() override = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to shared elements of the variable
 * in shared memory.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * This class works for vector mesh variables
 * with a maximum dimension of 2.
 *
 * If the mesh and/or the variable size changes when an object of this
 * type is used, it is necessary to call the \a updateVariable() method.
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMeshVectorMDVariableT
: public MachineShMemWinMDVariableT<ItemType, DataType, typename Extents::template AddedFirstExtentsType<DynExtent>>
{
  using AddedFirstExtentsType = Extents::template AddedFirstExtentsType<DynExtent>;

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  template <Int32 Size>
  explicit MachineShMemWinMeshVectorMDVariableT(MeshVectorMDVariableRefT<ItemType, DataType, Size, Extents> var)
  : MachineShMemWinMDVariableT<ItemType, DataType, AddedFirstExtentsType>(var.underlyingVariable())
  {}

  ~MachineShMemWinMeshVectorMDVariableT() override = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to shared elements of the variable
 * in shared memory.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * This class works for matrix mesh variables
 * with a maximum dimension of 1.
 *
 * If the mesh and/or the variable size changes when an object of this
 * type is used, it is necessary to call the \a updateVariable() method.
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMeshMatrixMDVariableT
: public MachineShMemWinMDVariableT<ItemType, DataType, typename Extents::template AddedFirstLastExtentsType<DynExtent, DynExtent>>
{
  using AddedFirstLastExtentsType = Extents::template AddedFirstLastExtentsType<DynExtent, DynExtent>;

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  template <Int32 Row, Int32 Column>
  explicit MachineShMemWinMeshMatrixMDVariableT(MeshMatrixMDVariableRefT<ItemType, DataType, Row, Column, Extents> var)
  : MachineShMemWinMDVariableT<ItemType, DataType, AddedFirstLastExtentsType>(var.underlyingVariable())
  {}

  ~MachineShMemWinMeshMatrixMDVariableT() override = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
