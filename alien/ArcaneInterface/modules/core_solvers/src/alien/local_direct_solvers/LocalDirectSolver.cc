
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <arccore/base/NotImplementedException.h>
#include <arccore/base/FatalErrorException.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include <arccore/message_passing/Communicator.h>
#include <arccore/trace/ITraceMng.h>

#include "alien/local_direct_solvers/algorithms/LUSolver.h"

#include "alien/expression/solver/ILinearSolver.h"
#include "alien/local_direct_solvers/ILinearSystem.h"
#include "alien/local_direct_solvers/ILinearSystemBuilder.h"
#include "alien/local_direct_solvers/IBaseLinearSolver.h"

#include "alien/local_direct_solvers/LocalDirectSolver.h"

using namespace Arcane;
using namespace std;
namespace Alien {

bool LocalDirectSolver::solve()
{
  if(!m_system_is_built)
  {
    cerr<<"Linear system is not built, buildLinearSystem shoud be called first"<<endl ;
    return  false ;
  }
  if(m_system_is_locked)
  {
    cerr<<"linear system has already be solved onece and has not been modified since" ;
    return false ;
  }
  LocalLinearSystem::RealMatrix& matrix = *(m_system->getMatrix()) ;
  LocalLinearSystem::RealVector& vk = *(m_system->getX()) ;
  LocalLinearSystem::RealVector& rk = *(m_system->getRhs()) ;
  try
  {
    LUSolver<Real> luJk;
    luJk.factor(matrix);
    vk = luJk.solve(-rk);
  }
  catch(LUSolver<Real>::Error & e)
  {
    m_status.succeeded = false ;
    m_status.error = 1 ;
    //m_problem->error() << e.msg;
    return false ;
  }
  catch(LUSolver<Real>::Warning & w)
  {
    //m_problem->warning() << w.msg;
    return false ;
  }
  m_status.succeeded = true ;
  m_status.error = 0 ;
  m_status.iteration_count = 0 ;
  m_status.residual = 0. ;
  m_system_is_locked = true ;
  return true ;
}
}

