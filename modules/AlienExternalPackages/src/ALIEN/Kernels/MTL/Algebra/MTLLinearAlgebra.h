#ifndef ALIEN_KERNELS_MTL_ALGEBRA_MTLLINEARALGEBRA_H
#define ALIEN_KERNELS_MTL_ALGEBRA_MTLLINEARALGEBRA_H

#include <ALIEN/Utils/Precomp.h>

#include <ALIEN/Kernels/MTL/MTLBackEnd.h>
#include <ALIEN/Core/Backend/LinearAlgebra.h>

namespace Alien {

typedef LinearAlgebra<BackEnd::tag::mtl> MTLLinearAlgebra;

}

#endif /* ALIEN_KERNELS_MTL_ALGEBRA_MTLLINEARALGEBRA_H */
