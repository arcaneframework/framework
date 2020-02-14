#ifndef ALIEN_KERNELS_PETSC_ALGEBRA_PETSCLINEARALGEBRA_H
#define ALIEN_KERNELS_PETSC_ALGEBRA_PETSCLINEARALGEBRA_H

#include <ALIEN/Utils/Precomp.h>

#include <ALIEN/Kernels/PETSc/PETScBackEnd.h>
#include <ALIEN/Core/Backend/LinearAlgebra.h>

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
