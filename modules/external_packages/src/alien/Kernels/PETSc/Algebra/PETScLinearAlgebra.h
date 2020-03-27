#ifndef ALIEN_KERNELS_PETSC_ALGEBRA_PETSCLINEARALGEBRA_H
#define ALIEN_KERNELS_PETSC_ALGEBRA_PETSCLINEARALGEBRA_H

#include <alien/utils/Precomp.h>

#include <alien/Kernels/PETSc/PETScBackEnd.h>
#include <alien/core/backend/LinearAlgebra.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef LinearAlgebra<BackEnd::tag::petsc> PETScLinearAlgebra;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNELS_PETSC_ALGEBRA_PETSCLINEARALGEBRA_H */
