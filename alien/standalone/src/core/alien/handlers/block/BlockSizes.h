// -*- C++ -*-
#pragma once

#include <alien/utils/Precomp.h>
#include <alien/utils/VMap.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IIndexManager;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT BlockSizes
{
 public:
  typedef VMap<Alien::Integer, Alien::Integer> ValuePerBlock;

  BlockSizes();

  ~BlockSizes() {}

  void prepare(const IIndexManager& index_mng, Alien::ConstArrayView<Alien::Integer> block_sizes);

  const ValuePerBlock& sizes() const { return m_sizes; }

 public:
  Alien::Integer size(Alien::Integer index) const;
  Alien::Integer sizeFromLocalIndex(Alien::Integer index) const;

  Alien::Integer offset(Alien::Integer index) const;
  Alien::Integer offsetFromLocalIndex(Alien::Integer index) const;

  Alien::Integer localSize() const;

  Alien::Integer maxSize() const;

  Alien::ConstArrayView<Alien::Integer> sizeOfLocalIndex() const;
  Alien::ConstArrayView<Alien::Integer> offsetOfLocalIndex() const;

 private:
  struct EntrySendRequest;
  struct EntryRecvRequest;

 private:
  bool m_is_prepared = false;
  Alien::IMessagePassingMng* m_parallel_mng = nullptr;

  Alien::Integer m_local_size = 0;
  Alien::Integer m_max_size = 0;

  ValuePerBlock m_sizes;
  ValuePerBlock m_offsets;

  SharedArray<Alien::Integer> m_local_sizes;
  SharedArray<Alien::Integer> m_local_offsets;
};
} // namespace Alien
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
