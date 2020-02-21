
#include "HPDDMInternalLinearAlgebra.h"

#include <ALIEN/Kernels/HPDDM/HPDDMBackEnd.h>

#include <ALIEN/Core/Backend/LinearAlgebraT.h>

#include <ALIEN/Data/Space.h>


#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRMatrix.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/Algebra/SimpleCSRInternalLinearAlgebra.h>

#include <arccore/base/NotImplementedException.h>
//#include <ALIEN/Kernels/HPDDM/DataStructure/HPDDMMatrix.h>
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

template class ALIEN_HPDDM_EXPORT LinearAlgebra<BackEnd::tag::hpddm> ;
//template class ALIEN_HPDDM_EXPORT LinearAlgebra<BackEnd::tag::hpddm,BackEnd::tag::simplecsr> ;

/*---------------------------------------------------------------------------*/
IInternalLinearAlgebra<SimpleCSRMatrix<Arccore::Real>, SimpleCSRVector<Arccore::Real>>*
HPDDMSolverInternalLinearAlgebraFactory()
{
  return new HPDDMSolverInternalLinearAlgebra();
}


} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
