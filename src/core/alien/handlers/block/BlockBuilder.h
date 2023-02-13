// -*- C++ -*-
#pragma once

#include <alien/utils/Precomp.h>
#include "alien/handlers/block/BlockSizes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IIndexManager;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT BlockBuilder
{
 public:
  class SizeVector
  {
   public:
    SizeVector(BlockBuilder& block_Builder, ConstArrayView<Integer> indexes);

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

  bool isLocal(Integer index) const
  {
    return (index >= m_offset) && (index < m_next_offset);
  }

 private:
  SharedArray<Integer> m_sizes;

  Integer m_offset = 0;
  Integer m_next_offset = 0;

  const IIndexManager& m_index_mng;
  mutable bool m_sizes_computed = false;
  mutable BlockSizes m_block_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien
