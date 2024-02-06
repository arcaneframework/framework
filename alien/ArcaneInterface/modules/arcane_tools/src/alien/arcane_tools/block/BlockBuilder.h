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

#pragma once

#include <alien/utils/Precomp.h>
#include <alien/AlienArcaneToolsExport.h>

#include "alien/arcane_tools/block/BlockSizes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

namespace ArcaneTools {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IIndexManager;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_ARCANE_TOOLS_EXPORT BlockBuilder
{
public:
  
  class ALIEN_ARCANE_TOOLS_EXPORT SizeVector
  {
  public:
    SizeVector(BlockBuilder& block_Builder, ConstArrayView<Integer>  indexes);
    
    void operator=(Integer size);
    void operator+=(Integer size);
    
  private:
    BlockBuilder& m_block_Builder;
    ConstArrayView<Integer> m_indexes;
  };

  BlockBuilder(IIndexManager& index_mng);
  
  ~BlockBuilder() {}
  
public:
  
  SizeVector operator[](ConstArrayView<Integer> indexes);

  Integer& operator[](Integer index);

  const BlockSizes::ValuePerBlock& sizes() const;

private:

  SharedArray<Integer> m_sizes;

  Integer m_offset = 0;

  const IIndexManager& m_index_mng;
  mutable bool m_sizes_computed = false;
  mutable BlockSizes m_block_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}
}
