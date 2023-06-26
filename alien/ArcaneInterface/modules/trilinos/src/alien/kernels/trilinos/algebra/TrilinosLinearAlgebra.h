/*
 * TrilinosLinearAlgebra.h
 *
 *  Created on: Jun 13, 2012
 *      Author: gratienj
 */
#ifndef ALIEN_KERNEL_TRILINOS_ALGEBRA_TRILINOSLINEARALGEBRA_H
#define ALIEN_KERNEL_TRILINOS_ALGEBRA_TRILINOSLINEARALGEBRA_H

#include <alien/utils/Precomp.h>

#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/core/backend/LinearAlgebra.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

#ifdef KOKKOS_USE_SERIAL
typedef LinearAlgebra<BackEnd::tag::tpetraserial> TrilinosLinearAlgebra;
#endif
#ifdef KOKKOS_USE_OPENMP
typedef LinearAlgebra<BackEnd::tag::tpetraomp> TrilinosLinearAlgebra;
typedef LinearAlgebra<BackEnd::tag::tpetraomp> TpetraOmpLinearAlgebra;
#endif
#ifdef KOKKOS_USE_THREADS
typedef LinearAlgebra<BackEnd::tag::tpetrapth> TpetraPthLinearAlgebra;
#endif
#ifdef KOKKOS_USE_CUDA
typedef LinearAlgebra<BackEnd::tag::tpetracuda> TpetraCudaLinearAlgebra;
#endif

} // namespace Alien

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNEL_TRILINOS_ALGEBRA_TRILINOSLINEARALGEBRA_H */
