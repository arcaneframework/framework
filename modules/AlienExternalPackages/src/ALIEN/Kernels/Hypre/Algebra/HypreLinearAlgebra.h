/*
 * HypreLinearAlgebra.h
 *
 *  Created on: Jun 13, 2012
 *      Author: gratienj
 */
#ifndef ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H
#define ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H

#include <ALIEN/Utils/Precomp.h>

#include <ALIEN/Kernels/Hypre/HypreBackEnd.h>
#include <ALIEN/Core/Backend/LinearAlgebra.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

typedef LinearAlgebra<BackEnd::tag::hypre> HypreLinearAlgebra;

}

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_LINEARALGEBRA2_HYPREIMPL_HYPRELINEARALGEBRA_H */
