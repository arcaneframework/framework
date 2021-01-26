#ifndef ALIEN_TRILINOSIMPL_TRILINOSBACKEND_H
#define ALIEN_TRILINOSIMPL_TRILINOSBACKEND_H
/* Author : mesriy at Tue Jul 24 15:56:45 2012
 * Generated by createNew
 */

#include <alien/utils/Precomp.h>
#include <alien/core/backend/BackEnd.h>
#include <Kokkos_Macros.hpp>
/*---------------------------------------------------------------------------*/
//#define KOKKOS_ENABLE_OPENMP
//#define KOKKOS_ENABLE_THREADS
//#define KOKKOS_ENABLE_CUDA

class IOptionsTrilinosSolver;

namespace Arccore::MessagePassing {
class IMessagePassingMng;
}

namespace Alien {

/*---------------------------------------------------------------------------*/

template <typename T, typename T2> class TrilinosMatrix;

template <typename T, typename T2> class TrilinosVector;

class Space;

template <class Matrix, class Vector> class IInternalLinearAlgebra;
template <class Matrix, class Vector> class IInternalLinearSolver;

/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct tpetraserial
    {
    };
    struct tpetraomp
    {
    };
    struct tpetrapth
    {
    };
    struct tpetracuda
    {
    };
  }
}
class Matrix;
class Vector;

extern IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetraserial>,
    TrilinosVector<Real, BackEnd::tag::tpetraserial>>*
TrilinosInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsTrilinosSolver* options);

extern IInternalLinearAlgebra<TrilinosMatrix<Real, BackEnd::tag::tpetraserial>,
    TrilinosVector<Real, BackEnd::tag::tpetraserial>>*
TrilinosInternalLinearAlgebraFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr);

template <> struct AlgebraTraits<BackEnd::tag::tpetraserial>
{
  typedef TrilinosMatrix<Real, BackEnd::tag::tpetraserial> matrix_type;
  typedef TrilinosVector<Real, BackEnd::tag::tpetraserial> vector_type;
  typedef IOptionsTrilinosSolver options_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;

  static algebra_type* algebra_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr)
  {
    return TrilinosInternalLinearAlgebraFactory();
  }

  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return TrilinosInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "tpetraserial"; }
};

#ifdef KOKKOS_ENABLE_OPENMP
extern IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetraomp>,
    TrilinosVector<Real, BackEnd::tag::tpetraomp>>*
TpetraOmpInternalLinearSolverFactory(
    IParallelMng* p_mng, IOptionsTrilinosSolver* options);

extern IInternalLinearAlgebra<TrilinosMatrix<Real, BackEnd::tag::tpetraomp>,
    TrilinosVector<Real, BackEnd::tag::tpetraomp>>*
TpetraOmpInternalLinearAlgebraFactory(IParallelMng* p_mng = nullptr);

template <> struct AlgebraTraits<BackEnd::tag::tpetraomp>
{
  typedef TrilinosMatrix<Real, BackEnd::tag::tpetraomp> matrix_type;
  typedef TrilinosVector<Real, BackEnd::tag::tpetraomp> vector_type;
  typedef IOptionsTrilinosSolver options_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;

  static algebra_type* algebra_factory(IParallelMng* p_mng = nullptr)
  {
    return TpetraOmpInternalLinearAlgebraFactory();
  }
  static solver_type* solver_factory(IParallelMng* p_mng, options_type* options)
  {
    return TpetraOmpInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "tpetraomp"; }
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
extern IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetrapth>,
    TrilinosVector<Real, BackEnd::tag::tpetrapth>>*
TpetraPthInternalLinearSolverFactory(
    IParallelMng* p_mng, IOptionsTrilinosSolver* options);

extern IInternalLinearAlgebra<TrilinosMatrix<Real, BackEnd::tag::tpetrapth>,
    TrilinosVector<Real, BackEnd::tag::tpetrapth>>*
TpetraPthInternalLinearAlgebraFactory(IParallelMng* p_mng = nullptr);

template <> struct AlgebraTraits<BackEnd::tag::tpetrapth>
{
  typedef TrilinosMatrix<Real, BackEnd::tag::tpetrapth> matrix_type;
  typedef TrilinosVector<Real, BackEnd::tag::tpetrapth> vector_type;
  typedef IOptionsTrilinosSolver options_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;

  static algebra_type* algebra_factory(IParallelMng* p_mng = nullptr)
  {
    return TpetraPthInternalLinearAlgebraFactory();
  }
  static solver_type* solver_factory(IParallelMng* p_mng, options_type* options)
  {
    return TpetraPthInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "tpetrapth"; }
};
#endif
/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_CUDA
extern IInternalLinearSolver<TrilinosMatrix<Real, BackEnd::tag::tpetracuda>,
    TrilinosVector<Real, BackEnd::tag::tpetracuda>>*
TpetraCudaInternalLinearSolverFactory(
    IParallelMng* p_mng, IOptionsTrilinosSolver* options);

extern IInternalLinearAlgebra<TrilinosMatrix<Real, BackEnd::tag::tpetracuda>,
    TrilinosVector<Real, BackEnd::tag::tpetracuda>>*
TpetraCudaInternalLinearAlgebraFactory(IParallelMng* p_mng = nullptr);

template <> struct AlgebraTraits<BackEnd::tag::tpetracuda>
{
  typedef TrilinosMatrix<Real, BackEnd::tag::tpetracuda> matrix_type;
  typedef TrilinosVector<Real, BackEnd::tag::tpetracuda> vector_type;
  typedef IOptionsTrilinosSolver options_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;

  static algebra_type* algebra_factory(IParallelMng* p_mng = nullptr)
  {
    return TpetraCudaInternalLinearAlgebraFactory();
  }
  static solver_type* solver_factory(IParallelMng* p_mng, options_type* options)
  {
    return TpetraCudaInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "tpetracuda"; }
};
#endif

} // namespace Alien

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_TRILINOSIMPL_TRILINOSBACKEND_H */
