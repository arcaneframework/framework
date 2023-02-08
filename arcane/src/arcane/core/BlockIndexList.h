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
 * \brief Classe gérant un tableau sour la forme d'une liste de blocs.
 * \warning Experimental API
 */
class ARCANE_CORE_EXPORT BlockIndexList
{
  friend class BlockIndexListBuilder;

  // TODO: ajouter une méthode shrinkMemory()

 public:

  Int32 nbBlock() const { return m_block_offsets.size(); }
  Real memoryRatio() const;
  void reset();

 private:

  UniqueArray<Int32> m_indexes;
  // Index dans 'm_indexes' de chaque bloc
  UniqueArray<Int32> m_block_indexes;
  // Valeur à ajouter pour chaque bloc.
  UniqueArray<Int32> m_block_offsets;
  Int32 m_original_size = 0;
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
