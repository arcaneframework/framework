#ifndef ALIEN_DISTRIBUTION_CREATEDISTRIBUTION_H
#define ALIEN_DISTRIBUTION_CREATEDISTRIBUTION_H

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/arcane_tools/IIndexManager.h>

#include <arcane/IParallelMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  inline VectorDistribution
  /*
  createVectorDistribution(IIndexManager* index_mng,
                           std::shared_ptr<IParallelMng> parallel_mng)
  */
  createVectorDistribution(IIndexManager* index_mng, Arcane::IParallelMng* parallel_mng)
  {
  auto global_size = index_mng->globalSize();
  auto local_size = index_mng->localSize();
  return VectorDistribution(global_size, local_size, parallel_mng->messagePassingMng());
  }

/*---------------------------------------------------------------------------*/

inline MatrixDistribution
/*
createMatrixDistribution(IIndexManager* index_mng,
                         std::shared_ptr<IParallelMng> parallel_mng)
*/
createMatrixDistribution(IIndexManager* index_mng, Arcane::IParallelMng* parallel_mng)
  {
  auto global_size = index_mng->globalSize();
  auto local_size = index_mng->localSize();
  return MatrixDistribution(
      global_size, global_size, local_size, parallel_mng->messagePassingMng());
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
