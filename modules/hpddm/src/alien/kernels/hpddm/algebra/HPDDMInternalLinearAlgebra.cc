
#include "HPDDMInternalLinearAlgebra.h"

#include <alien/kernels/hpddm/HPDDMBackEnd.h>

#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/data/Space.h>

#include <alien/kernels/simple_csr/data_structure/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#include <arccore/base/NotImplementedException.h>
//#include <alien/kernels/hpddm/data_structure/HPDDMMatrix.h>
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

template class ALIEN_HPDDM_EXPORT LinearAlgebra<BackEnd::tag::hpddm>;
// template class ALIEN_HPDDM_EXPORT
// LinearAlgebra<BackEnd::tag::hpddm,BackEnd::tag::simplecsr> ;

/*---------------------------------------------------------------------------*/
IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
HPDDMSolverInternalLinearAlgebraFactory()
{
  return new HPDDMSolverInternalLinearAlgebra();
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
