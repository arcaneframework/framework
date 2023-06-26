#pragma once

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/index_manager/IIndexManager.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

inline VectorDistribution
createVectorDistribution(
IIndexManager* index_mng, Alien::IMessagePassingMng* parallel_mng)
{
  auto global_size = index_mng->globalSize();
  auto local_size = index_mng->localSize();
  return VectorDistribution(global_size, local_size, parallel_mng);
}

/*---------------------------------------------------------------------------*/

inline MatrixDistribution
createMatrixDistribution(
IIndexManager* index_mng, Alien::IMessagePassingMng* parallel_mng)
{
  auto global_size = index_mng->globalSize();
  auto local_size = index_mng->localSize();
  return MatrixDistribution(global_size, global_size, local_size, parallel_mng);
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
