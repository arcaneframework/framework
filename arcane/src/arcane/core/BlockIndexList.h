// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BlockIndexList.h                                            (C) 2000-2025 */
/*                                                                           */
/* Class managing an array of indices in the form of a block list.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_BLOCKINDEXLIST_H
#define ARCANE_CORE_BLOCKINDEXLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/UniqueArray.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Block containing a list of indices with an offset.
 * \warning Experimental API
 */
class ARCANE_CORE_EXPORT BlockIndex
{
  friend class BlockIndexList;

 public:

  static constexpr Int16 MAX_BLOCK_SIZE = 512;

 private:

  BlockIndex(const Int32* ptr, Int32 value_offset, Int16 size)
  : m_block_start(ptr)
  , m_value_offset(value_offset)
  , m_size(size)
  {}

 public:

  //! i-th value of the block
  Int32 operator[](Int32 i) const
  {
    ARCANE_CHECK_AT(i, m_size);
    return m_block_start[i] + m_value_offset;
  }

  //! Size of the block
  Int16 size() const { return m_size; }

  //! Offset of the block values.
  Int32 valueOffset() const { return m_value_offset; }

 private:

  const Int32* m_block_start = nullptr;
  Int32 m_value_offset = 0;
  Int16 m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class managing an array in the form of a block list.
 * \warning Experimental API
 */
class ARCANE_CORE_EXPORT BlockIndexList
{
  friend class BlockIndexListBuilder;

  // TODO: be able to choose an allocator with accelerator support.

 public:

  Int32 nbBlock() const { return m_nb_block; }
  Real memoryRatio() const;
  void reset();
  BlockIndex block(Int32 i) const
  {
    Int32 index = m_blocks_index_and_offset[i * 2];
    Int32 offset = m_blocks_index_and_offset[(i * 2) + 1];
    Int16 size = ((i + 1) != m_nb_block) ? m_block_size : m_last_block_size;
    return BlockIndex(m_indexes.span().ptrAt(index), offset, size);
  }
  void fillArray(Array<Int32>& v);

 private:

  //! List of indexes
  UniqueArray<Int32> m_indexes;
  // Index in 'm_indexes' and offset for each block
  UniqueArray<Int32> m_blocks_index_and_offset;
  //! Original size of the index array
  Int32 m_original_size = 0;
  //! Number of blocks (m_original_size/m_block_size rounded up)
  Int32 m_nb_block = 0;
  //! Size of a block.
  Int16 m_block_size = 0;
  //! Size of the last block.
  Int16 m_last_block_size = 0;

 private:

  void _setBlockIndexAndOffset(Int32 block, Int32 index, Int32 offset);
  void _setNbBlock(Int32 nb_block);
  Int32 _currentIndexPosition() const;
  void _addBlockInfo(const Int32* data, Int16 size);
  Int32 _computeNbContigusBlock() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class allowing the construction of a BlockIndexList.
 * \warning Experimental API
 */
class ARCANE_CORE_EXPORT BlockIndexListBuilder
: public TraceAccessor
{
 public:

  BlockIndexListBuilder(ITraceMng* tm);

 public:

  void setVerbose(bool v) { m_is_verbose = v; }

  /*!
   * \brief Sets the block size as a power of 2.
   *
   * The size of a block will be equal to 2 ^ \a v.
   * If \a v==0, the block size is 1, if \a v==1, the size is 2,
   * if \a v==2 the size is 4, if \a v==3 the size is 8, and so on.
   */
  void setBlockSizeAsPowerOfTwo(Int32 v);

 public:

  void build(BlockIndexList& block_index_list, SmallSpan<const Int32> indexes, const String& name);

 private:

  bool m_is_verbose = false;
  Int16 m_block_size = 32;

 private:

  void _throwInvalidBlockSize [[noreturn]] (Int32 block_size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
