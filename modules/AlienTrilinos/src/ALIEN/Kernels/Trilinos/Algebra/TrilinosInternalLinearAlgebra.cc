


#include "ALIEN/Kernels/Trilinos/TrilinosPrecomp.h"
#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosInternal.h>

#include <ALIEN/Core/Backend/LinearAlgebraT.h>

#include <ALIEN/Data/Space.h>


#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosMatrix.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosVector.h>
#include <ALIEN/Kernels/Trilinos/Algebra/TrilinosInternalLinearAlgebra.h>

#include <arccore/base/NotImplementedException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
    template
    class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetraserial>;
IInternalLinearAlgebra<TrilinosMatrixType, TrilinosVectorType> *
TrilinosInternalLinearAlgebraFactory(Arccore::MessagePassing::IMessagePassingMng *pm)
{
  return new TrilinosInternalLinearAlgebra(pm);
}

#ifdef KOKKOS_ENABLE_OPENMP
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetraomp> ;

IInternalLinearAlgebra<TpetraOmpMatrixType, TpetraOmpVectorType>*
TpetraOmpInternalLinearAlgebraFactory(IParallelMng* pm)
{
  return new TpetraOmpInternalLinearAlgebra(pm);
}
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetrapth> ;

IInternalLinearAlgebra<TpetraPthMatrixType, TpetraPthVectorType>*
TpetraPthInternalLinearAlgebraFactory(IParallelMng* pm)
{
  return new TpetraPthInternalLinearAlgebra(pm);
}
#endif
/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_CUDA
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetracuda> ;

IInternalLinearAlgebra<TpetraCudaMatrixType, TpetraCudaVectorType>*
TpetraCudaInternalLinearAlgebraFactory(IParallelMng* pm)
{
  return new TpetraCudaInternalLinearAlgebra(pm);
}
#endif

/*---------------------------------------------------------------------------*/

    Arccore::Real
TrilinosInternalLinearAlgebra::norm0(const Vector& x) const
{
  return 0.;
}

/*---------------------------------------------------------------------------*/

    Arccore::Real
TrilinosInternalLinearAlgebra::norm1(const Vector& x) const
{
  return x.norm1() ;
}

/*---------------------------------------------------------------------------*/

    Arccore::Real
TrilinosInternalLinearAlgebra::norm2(const Vector& x) const
{
  return x.norm2() ;
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::mult(const Matrix& a, const Vector& x, Vector& r) const
{
  a.mult(x,r) ;
}

/*---------------------------------------------------------------------------*/

void
    TrilinosInternalLinearAlgebra::axpy(const Arccore::Real &alpha, const Vector &x, Vector &r) const {
        throw Arccore::NotImplementedException(
                A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
    }

/*---------------------------------------------------------------------------*/

void
    TrilinosInternalLinearAlgebra::aypx(const Arccore::Real &alpha, Vector &y, const Vector &x) const
{
    throw Arccore::NotImplementedException(
            A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::copy(const Vector& x, Vector& r) const {
        throw Arccore::NotImplementedException(
                A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
    }

/*---------------------------------------------------------------------------*/

    Arccore::Real
TrilinosInternalLinearAlgebra::dot(const Vector& x, const Vector& y) const
{
  return x.dot(y) ;
}

/*---------------------------------------------------------------------------*/

void
    TrilinosInternalLinearAlgebra::scal(const Arccore::Real &alpha, Vector &x) const
{
    throw Arccore::NotImplementedException(
            A_FUNCINFO, "TrilinosInternalLinearAlgebra::scal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::diagonal(const Matrix& a, Vector& x) const
{
    throw Arccore::NotImplementedException(
            A_FUNCINFO, "TrilinosInternalLinearAlgebra::diagonal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::reciprocal(Vector& x) const
{
    throw Arccore::NotImplementedException(
            A_FUNCINFO, "TrilinosInternalLinearAlgebra::reciprocal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::pointwiseMult(const Vector& x, const Vector& y, Vector& w) const
{
    throw Arccore::NotImplementedException(
            A_FUNCINFO, "TrilinosInternalLinearAlgebra::pointwiseMult not implemented");
}

void TrilinosInternalLinearAlgebra::dump(Matrix const& a,std::string const& filename) const
{
  a.dump(filename) ;
}

void TrilinosInternalLinearAlgebra::dump(Vector const& x,std::string const& filename) const
{
  x.dump(filename) ;
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
