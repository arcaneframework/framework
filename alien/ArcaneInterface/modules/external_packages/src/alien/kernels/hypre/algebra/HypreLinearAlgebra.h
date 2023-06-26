/*
 * HypreLinearAlgebra.h
 *
 *  Created on: Jun 13, 2012
 *      Author: gratienj
 */
#ifndef ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H
#define ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H

#include <alien/utils/Precomp.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/core/backend/LinearAlgebra.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

typedef LinearAlgebra<BackEnd::tag::hypre> HypreLinearAlgebra;
}

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H */
