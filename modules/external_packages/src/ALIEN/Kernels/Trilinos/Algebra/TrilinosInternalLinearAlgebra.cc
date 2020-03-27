


#include <Kokkos_Macros.hpp>

#include "ALIEN/Kernels/Trilinos/TrilinosPrecomp.h"
#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosInternal.h>

#include <alien/core/backend/LinearAlgebraT.h>
//#include <alien/kernels/simple_csr/algebra/CBLASMPIKernel.h>

#include <alien/data/Space.h>


#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosMatrix.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosVector.h>
#include <ALIEN/Kernels/Trilinos/Algebra/TrilinosInternalLinearAlgebra.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define KOKKOS_ENABLE_OPENMP
#define KOKKOS_ENABLE_THREADS

namespace Alien {

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
template class ALIEN_EXTERNAL_PACKAGES_EXPORT LinearAlgebra<BackEnd::tag::tpetraserial>;
IInternalLinearAlgebra<TrilinosMatrixType, TrilinosVectorType>*
TrilinosInternalLinearAlgebraFactory(IParallelMng* pm)
{
  return new TrilinosInternalLinearAlgebra(pm);
}

#ifdef KOKKOS_ENABLE_OPENMP
template class ALIEN_EXTERNAL_PACKAGES_EXPORT LinearAlgebra<BackEnd::tag::tpetraomp> ;

IInternalLinearAlgebra<TpetraOmpMatrixType, TpetraOmpVectorType>*
TpetraOmpInternalLinearAlgebraFactory(IParallelMng* pm)
{
  return new TpetraOmpInternalLinearAlgebra(pm);
}
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class ALIEN_EXTERNAL_PACKAGES_EXPORT LinearAlgebra<BackEnd::tag::tpetrapth> ;

IInternalLinearAlgebra<TpetraPthMatrixType, TpetraPthVectorType>*
TpetraPthInternalLinearAlgebraFactory(IParallelMng* pm)
{
  return new TpetraPthInternalLinearAlgebra(pm);
}
#endif
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

Real
TrilinosInternalLinearAlgebra::norm0(const Vector& x) const
{
  return 0.;
}

/*---------------------------------------------------------------------------*/

Real
TrilinosInternalLinearAlgebra::norm1(const Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

Real
TrilinosInternalLinearAlgebra::norm2(const Vector& x) const
{
	throw NotImplementedException(
	      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
	return 0. ;
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::mult(const Matrix& a, const Vector& x, Vector& r) const
{
  a.mult(x.getDataPtr(),r.getDataPtr()) ;
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::axpy(const Real& alpha, const Vector& x, Vector& r) const
{
	throw NotImplementedException(
	      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::aypx(const Real& alpha, Vector& y, const Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::copy(const Vector& x, Vector& r) const
{
	throw NotImplementedException(
	      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

Real
TrilinosInternalLinearAlgebra::dot(const Vector& x, const Vector& y) const
{
	throw NotImplementedException(
		      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
	return 0;
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::scal(const Real& alpha, Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::scal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::diagonal(const Matrix& a, Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::diagonal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::reciprocal(Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::reciprocal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::pointwiseMult(const Vector& x, const Vector& y, Vector& w) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::pointwiseMult not implemented");
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
