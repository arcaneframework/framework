// -*- C++ -*-
#ifndef ALIEN_USEROBJECTS_DATA_ARCANE_INDEXMANAGERBLOCKBUILDER_H
#define ALIEN_USEROBJECTS_DATA_ARCANE_INDEXMANAGERBLOCKBUILDER_H

#include <alien/core/block/IBlockBuilder.h>
#include <alien/utils/VMap.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>

#include "ALIEN/Alien-ArcaneToolsPrecomp.h"
#include "ALIEN/ArcaneTools/IIndexManager.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class IIndexManager;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_ARCANE_TOOLS_EXPORT IndexManagerBlockBuilder : public IBlockBuilder
{
public:
 typedef VMap<Arccore::Integer, Arccore::Integer> ValuePerBlock;

public:
  IndexManagerBlockBuilder(IIndexManager& index_mng,
                           const VectorDistribution& distribution);
  
  IndexManagerBlockBuilder(IIndexManager& index_mng,
                           const MatrixDistribution& distribution);

  ~IndexManagerBlockBuilder() {}

  void fill(Arccore::Integer value) { m_sizes.fill(value); }

  ValuePerBlock&& sizes() const { return std::move(ghost_sizes); }

  void compute() const;
private:

  struct EntryRecvRequest;
  struct EntrySendRequest;

private:
  IIndexManager* m_index_mng;
  mutable ValuePerBlock ghost_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_USEROBJECTS_DATA_ARCANE_INDEXMANAGERBLOCKBUILDER_H */
