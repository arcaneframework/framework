
#include "ComposyxInternalLinearAlgebra.h"

#include <alien/kernels/composyx/ComposyxBackEnd.h>

#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/data/Space.h>

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#include <arccore/base/NotImplementedException.h>
//#include <alien/kernels/composyx/data_structure/ComposyxMatrix.h>
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

//template class ALIEN_COMPOSYX_EXPORT LinearAlgebra<BackEnd::tag::composyx>;
// template class ALIEN_COMPOSYX_EXPORT
// LinearAlgebra<BackEnd::tag::composyx,BackEnd::tag::simplecsr> ;

/*---------------------------------------------------------------------------*/
IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
ComposyxSolverInternalLinearAlgebraFactory()
{
  return new ComposyxSolverInternalLinearAlgebra();
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
