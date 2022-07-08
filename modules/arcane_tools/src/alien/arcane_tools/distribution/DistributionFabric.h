#ifndef ALIEN_DISTRIBUTION_CREATEDISTRIBUTION_H
#define ALIEN_DISTRIBUTION_CREATEDISTRIBUTION_H

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/arcane_tools/IIndexManager.h>
#include <alien/arcane_tools/data/Space.h>

#include <arcane/IParallelMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  inline
  VectorDistribution
  createVectorDistribution(Space& space)
  {
    auto* im = space.indexManager();

    return Alien::VectorDistribution(space, im->localSize(), im->parallelMng());
  }

  inline VectorDistribution
  createVectorDistribution(IIndexManager* index_mng, Alien::IMessagePassingMng* parallel_mng)
  {
    auto global_size = index_mng->globalSize();
    auto local_size = index_mng->localSize();
    return VectorDistribution(global_size, local_size, parallel_mng);
  }

  /*---------------------------------------------------------------------------*/

  inline 
  VectorDistribution
  createVectorDistribution(Space& space, Arcane::Integer block_size)
  {
    auto* im = space.indexManager();
    
    return Alien::VectorDistribution(space, im->localSize()/block_size, im->parallelMng());
  }

  /*---------------------------------------------------------------------------*/
  inline MatrixDistribution
  createMatrixDistribution(IIndexManager* index_mng, Alien::IMessagePassingMng* parallel_mng)
  {
    auto global_size = index_mng->globalSize();
    auto local_size = index_mng->localSize();
    return MatrixDistribution(
        global_size, global_size, local_size, parallel_mng);
  }

  inline
  MatrixDistribution
  createMatrixDistribution(Space& space)
  {
    auto* im = space.indexManager();

    return Alien::MatrixDistribution(space, space, im->localSize(), im->parallelMng());
  }

  inline
  MatrixDistribution
  createMatrixDistribution(Space& row_space, Space& col_space)
  {
    auto* im = row_space.indexManager();

    return Alien::MatrixDistribution(row_space, col_space, im->localSize(), im->parallelMng());
  }

  /*---------------------------------------------------------------------------*/

  inline 
  MatrixDistribution
  createMatrixDistribution(Space& space, Arcane::Integer block_size)
  {
    auto* im = space.indexManager();
  
    return Alien::MatrixDistribution(space, space, im->localSize()/block_size, im->parallelMng());
  }
  
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_DISTRIBUTION_MATRIXDISTRIBUTION_H */
