// -*- C++ -*-
#ifndef ALIEN_KERNELS_TRILINOS_ALGEBRA_TRILINOSINTERNALLINEARALGEBRA_H
#define ALIEN_KERNELS_TRILINOS_ALGEBRA_TRILINOSINTERNALLINEARALGEBRA_H

#include <alien/AlienTrilinosPrecomp.h>
#include <alien/utils/Precomp.h>

#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef KOKKOS_ENABLE_SERIAL
typedef AlgebraTraits<BackEnd::tag::tpetraserial>::matrix_type TrilinosMatrixType;
typedef AlgebraTraits<BackEnd::tag::tpetraserial>::vector_type TrilinosVectorType;
#endif

class ALIEN_TRILINOS_EXPORT TrilinosInternalLinearAlgebra
    : public IInternalLinearAlgebra<TrilinosMatrixType, TrilinosVectorType>
{
 public:
  TrilinosInternalLinearAlgebra(Arccore::MessagePassing::IMessagePassingMng* pm = nullptr)
  {
  }

  virtual ~TrilinosInternalLinearAlgebra() {}

  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const;

  Real norm1(const Vector& x) const;

  Real norm2(const Vector& x) const;

  void mult(const Matrix& a, const Vector& x, Vector& r) const;

  void axpy(Real alpha, const Vector& x, Vector& r) const;

  void aypx(Real alpha, Vector& y, const Vector& x) const;

  void copy(const Vector& x, Vector& r) const;

  Real dot(const Vector& x, const Vector& y) const;

  void scal(Real alpha, Vector& x) const;

  void diagonal(const Matrix& a, Vector& x) const;

  void reciprocal(Vector& x) const;

  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;

  void mult(const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  void axpy(Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  void aypx(Real alpha, UniqueArray<Real>& y, const UniqueArray<Real>& x) const;
  void copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  Real dot(
      Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const;
  void scal(Real alpha, UniqueArray<Real>& x) const;

  void dump(Matrix const& a, std::string const& filename) const;
  void dump(Vector const& x, std::string const& filename) const;
};

#ifdef KOKKOS_ENABLE_OPENMP
typedef AlgebraTraits<BackEnd::tag::tpetraomp>::matrix_type TpetraOmpMatrixType;
typedef AlgebraTraits<BackEnd::tag::tpetraomp>::vector_type TpetraOmpVectorType;

class ALIEN_TRILINOS_EXPORT TpetraOmpInternalLinearAlgebra
    : public IInternalLinearAlgebra<TpetraOmpMatrixType, TpetraOmpVectorType>
{
 public:
  TpetraOmpInternalLinearAlgebra(Arccore::MessagePassing::IMessagePassingMng* pm = nullptr) {}
  virtual ~TpetraOmpInternalLinearAlgebra() {}

  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const { return 0.; }
  Real norm1(const Vector& x) const { return 0.; }
  Real norm2(const Vector& x) const { return 0.; }
  void mult(const Matrix& a, const Vector& x, Vector& r) const {}
  void axpy(Real alpha, const Vector& x, Vector& r) const {}
  void aypx(Real alpha, Vector& y, const Vector& x) const {}
  void copy(const Vector& x, Vector& r) const {}
  Real dot(const Vector& x, const Vector& y) const { return 0.; }
  void scal(Real alpha, Vector& x) const {}
  void diagonal(const Matrix& a, Vector& x) const {}
  void reciprocal(Vector& x) const {}
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const {}
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
typedef AlgebraTraits<BackEnd::tag::tpetrapth>::matrix_type TpetraPthMatrixType;
typedef AlgebraTraits<BackEnd::tag::tpetrapth>::vector_type TpetraPthVectorType;

class ALIEN_TRILINOS_EXPORT TpetraPthInternalLinearAlgebra
    : public IInternalLinearAlgebra<TpetraPthMatrixType, TpetraPthVectorType>
{
 public:
  TpetraPthInternalLinearAlgebra(Arccore::MessagePassing::IMessagePassingMng* pm = nullptr) {}
  virtual ~TpetraPthInternalLinearAlgebra() {}

  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const { return 0.; }
  Real norm1(const Vector& x) const { return 0.; }
  Real norm2(const Vector& x) const { return 0.; }
  void mult(const Matrix& a, const Vector& x, Vector& r) const {}
  void axpy(Real alpha, const Vector& x, Vector& r) const {}
  void aypx(Real alpha, Vector& y, const Vector& x) const {}
  void copy(const Vector& x, Vector& r) const {}
  Real dot(const Vector& x, const Vector& y) const { return 0.; }
  void scal(Real alpha, Vector& x) const {}
  void diagonal(const Matrix& a, Vector& x) const {}
  void reciprocal(Vector& x) const {}
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const {}
};
#endif

/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_CUDA
typedef AlgebraTraits<BackEnd::tag::tpetracuda>::matrix_type TpetraCudaMatrixType;
typedef AlgebraTraits<BackEnd::tag::tpetracuda>::vector_type TpetraCudaVectorType;

class ALIEN_TRILINOS_EXPORT TpetraCudaInternalLinearAlgebra
    : public IInternalLinearAlgebra<TpetraCudaMatrixType, TpetraCudaVectorType>
{
 public:
  TpetraCudaInternalLinearAlgebra(Arccore::MessagePassing::IMessagePassingMng* pm = nullptr) {}
  virtual ~TpetraCudaInternalLinearAlgebra() {}

  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const { return 0.; }
  Real norm1(const Vector& x) const { return 0.; }
  Real norm2(const Vector& x) const { return 0.; }
  void mult(const Matrix& a, const Vector& x, Vector& r) const {}
  void axpy(Real alpha, const Vector& x, Vector& r) const {}
  void aypx(Real alpha, Vector& y, const Vector& x) const {}
  void copy(const Vector& x, Vector& r) const {}
  Real dot(const Vector& x, const Vector& y) const { return 0.; }
  void scal(Real alpha, Vector& x) const {}
  void diagonal(const Matrix& a, Vector& x) const {}
  void reciprocal(Vector& x) const {}
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const {}
};
#endif

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_TRILINOSIMPL_TRILINOSLINEARALGEBRA_H */
