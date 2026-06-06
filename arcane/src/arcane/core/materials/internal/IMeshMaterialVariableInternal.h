// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableInternal.h                             (C) 2000-2024 */
/*                                                                           */
/* Arcane internal API for 'IMeshMaterialVariable'.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLEINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class ComponentItemListBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for copying between two memory regions.
 */
struct ARCANE_CORE_EXPORT CopyBetweenDataInfo
{
 public:

  CopyBetweenDataInfo(SmallSpan<const std::byte> input, SmallSpan<std::byte> output, Int32 data_size)
  : m_input(input)
  , m_output(output)
  , m_data_size(data_size)
  {}

 public:

  SmallSpan<const std::byte> m_input;
  SmallSpan<std::byte> m_output;
  Int32 m_data_size = 0;
};

/*!
 * \brief Common arguments for all MeshMaterialVariableIndexer methods.
 */
class ARCANE_CORE_EXPORT VariableIndexerCommonArgs
{
 public:

  VariableIndexerCommonArgs(Int32 var_index, const RunQueue& queue)
  : m_var_index(var_index)
  , m_queue(queue)
  {}

 public:

  void addOneCopyData(SmallSpan<const std::byte> input,
                      SmallSpan<std::byte> output,
                      Int32 data_size) const
  {
    if (m_copy_data) {
      CopyBetweenDataInfo x(input, output, data_size);
      m_copy_data->add(x);
    }
  }

  bool isUseOneCommand() const { return m_copy_data; }

 public:

  Int32 m_var_index = -1;
  RunQueue m_queue;
  //! Copy information if only one command is used
  UniqueArray<CopyBetweenDataInfo>* m_copy_data = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for methods copying between partial and global values
 */
class ARCANE_CORE_EXPORT CopyBetweenPartialAndGlobalArgs
: public VariableIndexerCommonArgs
{
 public:

  CopyBetweenPartialAndGlobalArgs(Int32 var_index,
                                  SmallSpan<const Int32> local_ids,
                                  SmallSpan<const Int32> indexes_in_multiple,
                                  bool do_copy,
                                  bool is_global_to_partial,
                                  const RunQueue& queue)
  : VariableIndexerCommonArgs(var_index, queue)
  , m_local_ids(local_ids)
  , m_indexes_in_multiple(indexes_in_multiple)
  , m_do_copy_between_partial_and_pure(do_copy)
  , m_is_global_to_partial(is_global_to_partial)
  {}

 public:

  SmallSpan<const Int32> m_local_ids;
  SmallSpan<const Int32> m_indexes_in_multiple;
  bool m_do_copy_between_partial_and_pure = true;
  bool m_is_global_to_partial = false;
  bool m_use_generic_copy = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for methods copying between partial and global values
 */
class ARCANE_CORE_EXPORT InitializeWithZeroArgs
: public VariableIndexerCommonArgs
{
 public:

  InitializeWithZeroArgs(Int32 var_index, SmallSpan<const Int32> indexes_in_multiple,
                         const RunQueue& queue)
  : VariableIndexerCommonArgs(var_index, queue)
  , m_indexes_in_multiple(indexes_in_multiple)
  {}

 public:

  SmallSpan<const Int32> m_indexes_in_multiple;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for methods copying between partial and global values
 */
class ARCANE_CORE_EXPORT ResizeVariableIndexerArgs
: public VariableIndexerCommonArgs
{
 public:

  ResizeVariableIndexerArgs(Int32 var_index, const RunQueue& queue)
  : VariableIndexerCommonArgs(var_index, queue)
  {}

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arcane internal API for 'IMeshMaterialVariable'.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableInternal
{
 public:

  virtual ~IMeshMaterialVariableInternal() = default;

 public:

  /*!
   *\brief Byte size to save a variable value.
   *
   * For a scalar variable, this is the size of the associated data type.
   * For an array variable, this is the size of the data type 
   * multiplied by the number of elements in the array.
   */
  virtual Int32 dataTypeSize() const = 0;

  /*!
   * \brief Copies the variable values into a buffer.
   *
   * \a queue may be null.
   */
  virtual void copyToBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
                            Span<std::byte> bytes, RunQueue* queue) const = 0;

  /*!
   * \brief Copies the variable values from a buffer.
   *
   * \a queue may be null.
   */
  virtual void copyFromBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
                              Span<const std::byte> bytes, RunQueue* queue) = 0;

  //! \internal
  virtual Ref<IData> internalCreateSaveDataRef(Integer nb_value) = 0;

  //! \internal
  virtual void saveData(IMeshComponent* component, IData* data) = 0;

  //! \internal
  virtual void restoreData(IMeshComponent* component, IData* data,
                           Integer data_index, Int32ConstArrayView ids, bool allow_null_id) = 0;

  //! \internal
  virtual void copyBetweenPartialAndGlobal(const CopyBetweenPartialAndGlobalArgs& args) = 0;

  //! Initialize the values of new components with zero
  virtual void initializeNewItemsWithZero(InitializeWithZeroArgs& args) = 0;

  //! List of 'VariableRef' associated with this variable.
  virtual ConstArrayView<VariableRef*> variableReferenceList() const = 0;

  //! Synchronizes references
  virtual void syncReferences(bool check_resize) = 0;

  //! Resizes the partial value associated with index \a index
  virtual void resizeForIndexer(ResizeVariableIndexerArgs& args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
