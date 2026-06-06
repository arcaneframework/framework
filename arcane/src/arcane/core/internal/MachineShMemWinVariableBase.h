// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariableBase.h                               (C) 2000-2026 */
/*                                                                           */
/* Base classes allowing the exploitation of the MachineShMemWinVariable     */
/* pointed to the shared memory variable zone.                               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_MACHINESHMEMWINVARIABLEBASE_H
#define ARCANE_CORE_INTERNAL_MACHINESHMEMWINVARIABLEBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/HashTableMap2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared memory part between
 * sub-domains of the same node of a variable.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariableBase
{

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableBase(IVariable* var);

 public:

  /*!
   * \brief Method allowing retrieval of the ranks that possess a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method allowing waiting until all processes/threads
   * of the node call this method to continue execution.
   */
  void barrier() const;

  /*!
   * \brief Method allowing retrieval of a view on the segment of another
   * sub-domain of the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the sub-domain.
   * \return A view.
   */
  Span<std::byte> segmentView(Int32 rank) const;

  /*!
   * \brief
   * \param nb_elem_dim1 In number of elements
   * \param sizeof_elem In bytes
   */
  void updateVariable(Int64 nb_elem_dim1, Int64 sizeof_elem);

 protected:

  IVariable* m_var = nullptr;
  IParallelMng* m_pm = nullptr;

  ConstArrayView<Int32> m_machine_ranks;

  // <world_rank, size_bytes>
  impl::HashTableMap2<Int32, Int64> m_sizeof_var;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared memory part between
 * sub-domains of the same node of a 2D array variable.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * In this class, the two dimensions can be different for each
 * sub-domain.
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariable2DBase
: public MachineShMemWinVariableBase
{

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariable2DBase(IVariable* var);

 public:

  /*!
   * \brief
   * \param nb_elem_dim1 In number of elements
   * \param nb_elem_dim2 In number of elements
   * \param sizeof_elem In bytes
   */
  void updateVariable(Int64 nb_elem_dim1, Int64 nb_elem_dim2, Int64 sizeof_elem);

  Int64 nbElemDim1(const Int32 rank) const { return m_nb_elem_dim1.at(rank); }
  Int64 nbElemDim2(const Int32 rank) const { return m_nb_elem_dim2.at(rank); }

 private:

  // <world_rank, size_nb_elems>
  impl::HashTableMap2<Int32, Int64> m_nb_elem_dim1;
  impl::HashTableMap2<Int32, Int64> m_nb_elem_dim2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing access to the shared memory part between
 * sub-domains of the same node of a 2D array variable.
 *
 * It is necessary that this variable be allocated in shared memory with
 * the property "IVariable::PInShMem".
 *
 * In this class, only the first dimension can be different for
 * each sub-domain. The second dimension must be the same for all
 * sub-domains.
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariableMDBase
: public MachineShMemWinVariableBase
{

 public:

  /*!
   * \brief Constructor.
   * \param var Variable having the property "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableMDBase(IVariable* var);

 public:

  /*!
   * \brief
   * \param nb_elem_dim1 In number of elements
   * \param nb_elem_dim2 In number of elements
   * \param sizeof_elem In bytes
   */
  void updateVariable(Int64 nb_elem_dim1, Int32 nb_elem_dim2, Int64 sizeof_elem);

  Int64 nbElemDim1(const Int32 rank) const { return m_nb_elem_dim1.at(rank); }
  ArrayShape arrayShape() const;

 private:

  // <world_rank, size_nb_elems>
  impl::HashTableMap2<Int32, Int64> m_nb_elem_dim1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
