// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BlockBuilder                                   (C) 2000-2024              */
/*                                                                           */
/* Builder for block matrices                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/utils/Precomp.h>
#include "alien/arcane_tools/IIndexManager.h"
#include "BlockBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

namespace ArcaneTools {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockBuilder::SizeVector::
SizeVector(BlockBuilder& block_Builder, ConstArrayView<Integer> indexes)
  : m_block_Builder(block_Builder)
  , m_indexes(indexes) {}

/*---------------------------------------------------------------------------*/

void 
BlockBuilder::SizeVector::
operator=(Integer size)
{
  for(Integer i = 0; i < m_indexes.size(); ++i)
    m_block_Builder[m_indexes[i]] = size;
}

/*---------------------------------------------------------------------------*/

void 
BlockBuilder::SizeVector::
operator+=(Integer size)
{
  for(Integer i = 0; i < m_indexes.size(); ++i)
    m_block_Builder[m_indexes[i]] += size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockBuilder::
BlockBuilder(IIndexManager& index_mng)
  : m_offset(0)
  , m_index_mng(index_mng)
  , m_sizes_computed(false)
{
  if(not index_mng.isPrepared()) index_mng.prepare();

  Integer global_size, local_size;
  index_mng.stats(global_size, m_offset, local_size);
  
  m_sizes.resize(local_size);
}

/*---------------------------------------------------------------------------*/

BlockBuilder::SizeVector 
BlockBuilder::
operator[](ConstArrayView<Integer>  indexes)
{
  return SizeVector(*this, indexes);
}

/*---------------------------------------------------------------------------*/

Integer& 
BlockBuilder::
operator[](Integer index)
{
  return m_sizes[index-m_offset];
}

/*---------------------------------------------------------------------------*/

const BlockSizes::ValuePerBlock&
BlockBuilder::
sizes() const
{
  if(m_sizes_computed)
    return m_block_sizes.sizes();

  m_block_sizes.prepare(m_index_mng, m_sizes);

  m_sizes_computed = true;

  return m_block_sizes.sizes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}
}

