/*
 * HypreLinearAlgebra.h
 *
 *  Created on: Jun 13, 2012
 *      Author: gratienj
 */
#ifndef ALIEN_KERNEL_MCG_ALGEBRA_MCGLINEARALGEBRA_H
#define ALIEN_KERNEL_MCG_ALGEBRA_MCGLINEARALGEBRA_H

#include <alien/utils/Precomp.h>

#include <alien/kernels/mcg/MCGBackEnd.h>
#include <alien/core/backend/LinearAlgebra.h>
#include <alien/kernels/mcg/algebra/MCGInternalLinearAlgebra.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

// typedef LinearAlgebra<BackEnd::tag::mcgsolver> MCGLinearAlgebra;
typedef MCGInternalLinearAlgebra MCGLinearAlgebra;

} // namespace Alien

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H */
