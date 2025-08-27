// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BlockIndexList.h                                            (C) 2000-2023 */
/*                                                                           */
/* Classe gérant un tableau d'indices sous la forme d'une liste de blocs.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_BLOCKINDEXLIST_H
#define ARCANE_BLOCKINDEXLIST_H
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
 * \brief Bloc contenant une une liste d'indices avec un offset.
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

  //! i-ème valeur du bloc
  Int32 operator[](Int32 i) const
  {
    ARCANE_CHECK_AT(i, m_size);
    return m_block_start[i] + m_value_offset;
  }

  //! Taille du bloc
  Int16 size() const { return m_size; }

  //! Offset des valeurs du bloc.
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
 * \brief Classe gérant un tableau sous la forme d'une liste de blocs.
 * \warning Experimental API
 */
class ARCANE_CORE_EXPORT BlockIndexList
{
  friend class BlockIndexListBuilder;

  // TODO: pouvoir choisir un allocateur avec support accélérateur.

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

  //! Liste des indexes
  UniqueArray<Int32> m_indexes;
  // Index dans 'm_indexes' et offset de chaque bloc
  UniqueArray<Int32> m_blocks_index_and_offset;
  //! Taille d'origine du tableau d'indices
  Int32 m_original_size = 0;
  //! Nombre de blocs (m_original_size/m_block_size arrondi au supérieur)
  Int32 m_nb_block = 0;
  //! Taille  d'un bloc.
  Int16 m_block_size = 0;
  //! Taille du dernier bloc.
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
 * \brief Classe permettant de construire un BlockIndexList.
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
   * \brief Positionne la taille de bloc sous la forme d'une puissance de 2.
   *
   * La taille d'un bloc sera égal à 2 ^ \a v.
   * Si \a v==0, la taille du bloc est de 1, si \a v==1, la taille est de 2,
   * si \a v==2 la taille est de 4, si \a v==3 la taille est de 8 et ainsi de suite.
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
