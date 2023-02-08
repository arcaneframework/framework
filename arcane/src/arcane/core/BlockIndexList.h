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

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

  // TODO: ajouter une méthode shrinkMemory()
  // TODO: utiliser un seul tableau pour m_indexes, m_block_indexes et m_block_offsets
  // TODO: pouvoir choisir un allocateur avec support accélérateur.

  struct BlockIndex
  {
    friend class BlockIndexList;

   private:

    BlockIndex(const Int32* ptr, Int32 size, Int32 offset)
    : m_block_start(ptr)
    , m_offset(offset)
    , m_size(size)
    {}

   public:

    Int32 operator[](Int32 i) const
    {
      ARCANE_CHECK_AT(i, m_size);
      return m_block_start[i] + m_offset;
    }
    Int32 size() const { return m_size; }

   private:

    const Int32* m_block_start;
    Int32 m_offset;
    Int32 m_size;
  };

 public:

  Int32 nbBlock() const { return m_block_offsets.size(); }
  Real memoryRatio() const;
  void reset();
  BlockIndex block(Int32 i) const
  {
    Int32 idx = m_block_indexes[i];
    Int32 size = ((i + 1) != m_nb_block) ? m_block_size : m_last_block_size;
    return BlockIndex(m_indexes.span().ptrAt(idx), size, m_block_offsets[i]);
  }
  void fillArray(Array<Int32>& v);

 private:

  UniqueArray<Int32> m_indexes;
  // Index dans 'm_indexes' de chaque bloc
  UniqueArray<Int32> m_block_indexes;
  // Valeur à ajouter pour chaque bloc.
  UniqueArray<Int32> m_block_offsets;
  Int32 m_original_size = 0;
  Int32 m_block_size = 0;
  Int32 m_nb_block = 0;
  Int32 m_last_block_size = 0;
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
  // TODO: Ne supporter que des tailles de bloc qui sont des puissances de 2

 public:

  BlockIndexListBuilder(ITraceMng* tm);

 public:

  void setVerbose(bool v) { m_is_verbose = v; }
  void setBlockSize(Int32 v) { m_block_size = v; }

 public:

  void build(BlockIndexList& block_index_list, SmallSpan<const Int32> indexes, const String& name);

 private:

  bool m_is_verbose = false;
  Int32 m_block_size = 32;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
