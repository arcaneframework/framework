#ifndef ALIEN_KERNELS_MTL_ALGEBRA_MTLLINEARALGEBRA_H
#define ALIEN_KERNELS_MTL_ALGEBRA_MTLLINEARALGEBRA_H

#include <alien/utils/Precomp.h>

#include <alien/kernels/mtl/MTLBackEnd.h>
#include <alien/core/backend/LinearAlgebra.h>

namespace Alien {

typedef LinearAlgebra<BackEnd::tag::mtl> MTLLinearAlgebra;
}

#endif /* ALIEN_KERNELS_MTL_ALGEBRA_MTLLINEARALGEBRA_H */
