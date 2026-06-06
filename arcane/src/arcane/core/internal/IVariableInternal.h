// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IvariableInternal.h                                         (C) 2000-2025 */
/*                                                                           */
/* Internal part of IVariable in Arcane.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IVARIABLEINTERNAL_H
#define ARCANE_CORE_INTERNAL_IVARIABLEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for resizing a variable.
 */
class VariableResizeArgs
{
 public:

  explicit VariableResizeArgs(Int32 new_size)
  : m_new_size(new_size)
  {
  }

  explicit VariableResizeArgs(Int32 new_size, Int32 additional_capacity)
  : m_new_size(new_size)
  , m_additional_capacity(additional_capacity)
  {
  }

  explicit VariableResizeArgs(Int32 new_size, Int32 additional_capacity, bool use_no_init)
  : m_new_size(new_size)
  , m_additional_capacity(additional_capacity)
  , m_is_use_no_init(use_no_init)
  {
  }

  Int32 newSize() const { return m_new_size; }
  Int32 nbAdditionalCapacity() const { return m_additional_capacity; }
  bool isUseNoInit() const { return m_is_use_no_init; }

 private:

  Int32 m_new_size = 0;
  Int32 m_additional_capacity = 0;
  bool m_is_use_no_init = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal part of Ivariable.
 */
class ARCANE_CORE_EXPORT IVariableInternal
{
 public:

  virtual ~IVariableInternal() = default;

 public:

  /*!
   * \brief Calculates the comparison hash for the variable.
   *
   * \a sorted_data must be sorted according to uniqueId() and also
   * by rank of the IParallelMng associated with the variable.
   *
   * This method is collective, but only the master rank (the one for which
   * IParallelMng::isMasterIO() is true) returns a valid hash. The others
   * return a null string.
   *
   * It also returns a null string if the data is not numeric
   * (if sorted_data->_commonInternal()->numericData()==nullptr) or if
   * the variable is not associated with a mesh entity.
   */
  virtual String computeComparisonHashCollective(IHashAlgorithm* hash_algo,
                                                 IData* sorted_data) = 0;

  /*!
   * \brief Changes the variable's allocator.
   *
   * Currently valid only for 1D variables. Does nothing for others.
   *
   * \warning For experimental use only.
   */
  virtual void changeAllocator(const MemoryAllocationOptions& alloc_info) = 0;

  //! Resizes the variable by adding additional capacity
  virtual void resize(const VariableResizeArgs& resize_args) = 0;

  //! Applies the comparison method specified by \a compare_args
  virtual VariableComparerResults compareVariable(const VariableComparerArgs& compare_args) = 0;

  /*!
   * \brief Returns the IParallelMng of the mesh replica associated with the variable.
   *
   * Returns nullptr if there is no replication.
   */
  virtual IParallelMng* replicaParallelMng() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
