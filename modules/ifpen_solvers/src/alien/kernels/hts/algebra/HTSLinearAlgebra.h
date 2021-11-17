/*
 * HTSLinearAlgebra.h
 *
 *  Created on: Jun 13, 2012
 *      Author: gratienj
 */
#ifndef ALIEN_KERNEL_HTS_ALGEBRA_HTSLINEARALGEBRA_H
#define ALIEN_KERNEL_HTS_ALGEBRA_HTSLINEARALGEBRA_H

#include <alien/utils/Precomp.h>

#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/core/backend/LinearAlgebra.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

typedef LinearAlgebra<BackEnd::tag::htssolver> HTSSolverLinearAlgebra;
typedef LinearAlgebra<BackEnd::tag::hts, BackEnd::tag::simplecsr> HTSLinearAlgebra;

} // namespace Alien

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H */
