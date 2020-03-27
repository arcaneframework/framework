/*
 * TrilinosLinearAlgebra.h
 *
 *  Created on: Jun 13, 2012
 *      Author: gratienj
 */
#ifndef ALIEN_KERNEL_TRILINOS_ALGEBRA_TRILINOSLINEARALGEBRA_H
#define ALIEN_KERNEL_TRILINOS_ALGEBRA_TRILINOSLINEARALGEBRA_H

#include <alien/utils/Precomp.h>

#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <alien/core/backend/LinearAlgebra.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

typedef LinearAlgebra<BackEnd::tag::tpetraserial> TrilinosLinearAlgebra;
typedef LinearAlgebra<BackEnd::tag::tpetraomp> TpetraOmpLinearAlgebra;
typedef LinearAlgebra<BackEnd::tag::tpetrapth> TpetraPthLinearAlgebra;

} // namespace Alien

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H */
