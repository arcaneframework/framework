/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * MVExpr.h
 *
 *  Created on: Sep 30, 2019
 *      Author: gratienj
 */

#ifndef ALIEN_EXPRESSION_MVEXPR_MVEXPR_H_
#define ALIEN_EXPRESSION_MVEXPR_MVEXPR_H_

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Matrix;
class Vector;

namespace MVExpr
{
  namespace lazy
  {
    struct add_tag
    {};
    struct minus_tag
    {};
    struct mult_tag
    {};
    struct div_tag
    {};
    struct dot_tag
    {};
    struct cst_tag
    {};
    struct ref_tag
    {};
    struct assign_tag
    {};
    struct eval_tag
    {};
  } // namespace lazy

  template <class A, class B>
  auto add(A&& a, B&& b)
  {
    return [=](auto visitor) { return visitor(lazy::add_tag{}, a(visitor), b(visitor)); };
  }

  template <class A, class B>
  auto minus(A&& a, B&& b)
  {
    return
    [=](auto visitor) { return visitor(lazy::minus_tag{}, a(visitor), b(visitor)); };
  }

  template <class A, class B>
  auto mul(A&& a, B&& b)
  {
    return
    [=](auto visitor) { return visitor(lazy::mult_tag{}, a(visitor), b(visitor)); };
  }

  template <class A, class B>
  auto div(A&& a, B&& b)
  {
    return [=](auto visitor) { return visitor(lazy::div_tag{}, a(visitor), b(visitor)); };
  }

  struct distribution_evaluator;
  template <class A, class B>
  auto scalMul(A&& a, B&& b)
  {
    return [=](auto visitor) {
      return visitor(
      lazy::dot_tag{}, a(distribution_evaluator()), a(visitor), b(visitor));
    };
  }

  template <class T>
  auto cst(T expr)
  {
    return [=](auto visitor) { return visitor(lazy::cst_tag{}, expr); };
  }

  template <class T>
  auto ref(T const& expr)
  {
    return [&](auto visitor) -> decltype(visitor(lazy::ref_tag{}, expr)) {
      return visitor(lazy::ref_tag{}, expr);
    };
  }

  struct cpu_evaluator;
  struct alloc_size_evaluator;

  template <typename T>
  auto matrixMult(Matrix const& matrix, UniqueArray<T> const& x)
  {
#ifdef DEBUG
    std::cout << "\t\t MatrixVectorMult" << std::endl;
#endif
    std::size_t n = matrix.distribution().localRowSize();
    UniqueArray<T> y(n, 0.);
    SimpleCSRLinearAlgebraExpr alg;
    alg.mult(matrix, x, y);
    return std::move(y);
  }

  template <typename Tag, typename T>
  auto matrixMultT(Matrix const& matrix, UniqueArray<T> const& x)
  {
    std::size_t n = matrix.distribution().localRowSize();
    UniqueArray<T> y(n, 0.);
    LinearAlgebraExpr<Tag> alg(matrix.distribution().parallelMng());
    alg.mult(matrix, x, y);
    return std::move(y);
  }

  template <typename T>
  auto vectorAdd(UniqueArray<T> const& x, UniqueArray<T> const& y)
  {
#ifdef DEBUG
    std::cout << "\t\t VectorAdd" << std::endl;
#endif
    std::size_t n = x.size();
    UniqueArray<T> result(n, 0.);
    SimpleCSRLinearAlgebraExpr alg;
    alg.copy(y, result);
    alg.axpy(1., x, result);
    return std::move(result);
  }

  template <typename Tag, typename T>
  auto vectorAddT(UniqueArray<T> const& x, UniqueArray<T> const& y)
  {
    std::size_t n = x.size();
    UniqueArray<T> result(n, 0.);
    LinearAlgebraExpr<Tag> alg(nullptr);
    alg.copy(y, result);
    alg.axpy(1., x, result);
    return std::move(result);
  }

  template <typename Tag>
  auto matrixAddT(Matrix const& a, Matrix const& b)
  {
    LinearAlgebraExpr<Tag> alg(a.distribution().parallelMng());
    Matrix c(a.distribution());
    alg.copy(b, c);
    alg.add(a, c);
    return std::move(c);
  }

  //template <typename T>
  auto matrixAdd(Matrix const& a, Matrix const& b)
  {
#ifdef DEBUG
    std::cout << "\t\t MatrixAdd" << std::endl;
#endif

    Matrix c(a.distribution());
    SimpleCSRLinearAlgebraExpr alg;
    alg.copy(b, c);
    alg.add(a, c);
    return std::move(c);
  }

  template <typename T>
  auto vectorMinus(UniqueArray<T> const& x, UniqueArray<T> const& y)
  {
#ifdef DEBUG
    std::cout << "\t\t VectorMinus" << std::endl;
#endif
    std::size_t n = x.size();
    UniqueArray<T> result(n, 0.);

    SimpleCSRLinearAlgebraExpr alg;
    alg.copy(x, result);
    alg.axpy(-1., y, result);
    return std::move(result);
  }

  template <typename Tag, typename T>
  auto vectorMinusT(UniqueArray<T> const& x, UniqueArray<T> const& y)
  {
    std::size_t n = x.size();
    UniqueArray<T> result(n, 0.);

    LinearAlgebraExpr<Tag> alg(nullptr);
    alg.copy(x, result);
    alg.axpy(-1., y, result);
    return std::move(result);
  }

  template <typename T>
  auto vectorMult(T const& lambda, UniqueArray<T> const& x)
  {
#ifdef DEBUG
    std::cout << "\t\t VectorScal" << std::endl;
#endif
    std::size_t n = x.size();
    UniqueArray<T> y(n, 0.);
    SimpleCSRLinearAlgebraExpr alg;
    alg.axpy(lambda, x, y);
    return std::move(y);
  }

  template <typename Tag, typename T>
  auto vectorMultT(T const& lambda, UniqueArray<T> const& x)
  {
    std::size_t n = x.size();
    UniqueArray<T> y(n, 0.);
    LinearAlgebraExpr<Tag> alg(nullptr);
    alg.axpy(lambda, x, y);
    return std::move(y);
  }

  auto vectorScalProduct(Vector const& a, Vector const& b)
  {
#ifdef DEBUG
    std::cout << "\t\t VectorScal" << std::endl;
#endif
    SimpleCSRLinearAlgebraExpr alg;
    return alg.dot(a, b);
  }

  template <typename Tag>
  auto vectorScalProductT(Vector const& a, Vector const& b)
  {
    LinearAlgebraExpr<Tag> alg(a.distribution().parallelMng());
    return alg.dot(a, b);
  }

  auto vectorScalProduct(
  VectorDistribution const* distribution, Vector const& a, UniqueArray<Real> const& b)
  {
#ifdef DEBUG
    std::cout << "\t\t VectorScal" << std::endl;
#endif
    SimpleCSRVector<Real> const& csr_a = a.impl()->get<BackEnd::tag::simplecsr>();
    SimpleCSRLinearAlgebraExpr alg;
    Integer local_size = distribution ? distribution->localSize() : b.size();
    Real value = alg.dot(local_size, csr_a.getArrayValues(), b);
    if (distribution && distribution->isParallel())
      return Arccore::MessagePassing::mpAllReduce(
      distribution->parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    else
      return value;
  }

  template <typename Tag>
  auto vectorScalProductT(
  VectorDistribution const* distribution, Vector const& a, UniqueArray<Real> const& b)
  {
    auto const& csr_a = a.impl()->get<Tag>();
    LinearAlgebraExpr<Tag> alg(nullptr);
    Integer local_size = distribution ? distribution->localSize() : b.size();
    Real value = alg.dot(local_size, csr_a.getArrayValues(), b);
    if (distribution && distribution->isParallel())
      return Arccore::MessagePassing::mpAllReduce(
      distribution->parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    else
      return value;
  }

  auto vectorScalProduct(
  VectorDistribution const* distribution, UniqueArray<Real> const& a, Vector const& b)
  {
#ifdef DEBUG
    std::cout << "\t\t VectorScal" << std::endl;
#endif
    SimpleCSRVector<Real> const& csr_b = b.impl()->get<BackEnd::tag::simplecsr>();
    SimpleCSRLinearAlgebraExpr alg;
    Integer local_size = distribution ? distribution->localSize() : a.size();
    Real value = alg.dot(local_size, a, csr_b.getArrayValues());
    if (distribution && distribution->isParallel())
      return Arccore::MessagePassing::mpAllReduce(
      distribution->parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    else
      return value;
  }

  template <typename Tag>
  auto vectorScalProductT(
  VectorDistribution const* distribution, UniqueArray<Real> const& a, Vector const& b)
  {
    auto const& csr_b = b.impl()->get<Tag>();
    LinearAlgebraExpr<Tag> alg(nullptr);
    Integer local_size = distribution ? distribution->localSize() : a.size();
    Real value = alg.dot(local_size, a, csr_b.getArrayValues());
    if (distribution && distribution->isParallel())
      return Arccore::MessagePassing::mpAllReduce(
      distribution->parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    else
      return value;
  }

  auto vectorScalProduct(VectorDistribution const* distribution,
                         UniqueArray<Real> const& a, UniqueArray<Real> const& b)
  {
#ifdef DEBUG
    std::cout << "\t\t VectorScal" << std::endl;
#endif
    SimpleCSRLinearAlgebraExpr alg;
    Integer local_size = distribution ? distribution->localSize() : a.size();
    Real value = alg.dot(local_size, a, b);
    if (distribution && distribution->isParallel())
      return Arccore::MessagePassing::mpAllReduce(
      distribution->parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    else
      return value;
  }

  template <typename Tag>
  auto vectorScalProductT(VectorDistribution const* distribution,
                          UniqueArray<Real> const& a, UniqueArray<Real> const& b)
  {
    LinearAlgebraExpr<Tag> alg(distribution->parallelMng());
    Integer local_size = distribution ? distribution->localSize() : a.size();
    Real value = alg.dot(local_size, a, b);
    if (distribution && distribution->isParallel())
      return Arccore::MessagePassing::mpAllReduce(
      distribution->parallelMng(), Arccore::MessagePassing::ReduceSum, value);
    else
      return value;
  }

  template <typename Tag, typename T>
  auto matrixScalT(T const& lambda, Matrix const& A)
  {
    Matrix B(A.distribution());
    LinearAlgebraExpr<Tag> alg(A.distribution().parallelMng());
    alg.copy(A, B);
    alg.scal(lambda, B);
    return std::move(B);
  }

  template <typename T>
  auto matrixScal(T const& lambda, Matrix const& A)
  {
#ifdef DEBUG
    std::cout << "\t\t MatrixScal" << std::endl;
#endif
    Matrix B(A.distribution());
    SimpleCSRLinearAlgebraExpr alg;
    alg.copy(A, B);
    alg.scal(lambda, B);
    return std::move(B);
  }

  struct cpu_evaluator
  {

    template <typename T>
    auto operator()(lazy::cst_tag, T c)
    {
#ifdef DEBUG
      std::cout << "\t return cst" << std::endl;
#endif
      return c;
    }

    template <typename T>
    T const& operator()(lazy::ref_tag, T const& r)
    {
#ifdef DEBUG
      std::cout << "\t return ref : " << r.name() << std::endl;
#endif
      return r;
    }

    template <typename T>
    auto operator()(lazy::mult_tag, Matrix const& a, UniqueArray<T> const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit A*b" << std::endl;
#endif
      return matrixMult(a, b);
    }

    // template<typename T>
    auto operator()(lazy::mult_tag, Matrix const& a, Vector const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit A*b" << std::endl;
#endif
      SimpleCSRMatrix<Real> const& csr_matrix = a.impl()->get<BackEnd::tag::simplecsr>();
      SimpleCSRVector<Real> const& csr_b = b.impl()->get<BackEnd::tag::simplecsr>();
      csr_b.resize(csr_matrix.getAllocSize());
      return matrixMult(a, csr_b.getArrayValues());
    }

    auto operator()(lazy::mult_tag, Real lambda, Vector const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit lambda*b : " << b.name() << std::endl;
#endif
      SimpleCSRVector<Real> const& csr_b = b.impl()->get<BackEnd::tag::simplecsr>();
      return vectorMult(lambda, csr_b.getArrayValues());
    }

    auto operator()(lazy::mult_tag, Real lambda, Matrix const& a)
    {
#ifdef DEBUG
      std::cout << "\t visit lambda*b : " << b.name() << std::endl;
#endif
      return matrixScal(lambda, a);
    }

    auto operator()(lazy::add_tag, Vector const& a, Vector const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit a+b" << std::endl;
#endif
      SimpleCSRVector<Real> const& csr_a = a.impl()->get<BackEnd::tag::simplecsr>();
      SimpleCSRVector<Real> const& csr_b = b.impl()->get<BackEnd::tag::simplecsr>();
      return vectorAdd(csr_a.getArrayValues(), csr_b.getArrayValues());
    }

    auto operator()(lazy::add_tag, Vector const& a, UniqueArray<Real> const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit a+b" << std::endl;
#endif
      SimpleCSRVector<Real> const& csr_a = a.impl()->get<BackEnd::tag::simplecsr>();
      return vectorAdd(csr_a.getArrayValues(), b);
    }

    template <class T>
    auto operator()(lazy::add_tag, UniqueArray<T> const& a, UniqueArray<T> const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit a+b" << std::endl;
#endif
      return vectorAdd(a, b);
    }

    auto operator()(lazy::add_tag, Matrix const& a, Matrix const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit a+b" << std::endl;
#endif
      return matrixAdd(a, b);
    }

    // template<class A, class B>
    // auto operator()(lazy::add_tag, A a, B b) { return a + b; }

    auto operator()(lazy::minus_tag, Vector const& a, Vector const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit a-b" << std::endl;
#endif
      SimpleCSRVector<Real> const& csr_a = a.impl()->get<BackEnd::tag::simplecsr>();
      SimpleCSRVector<Real> const& csr_b = b.impl()->get<BackEnd::tag::simplecsr>();
      return vectorMinus(csr_a.getArrayValues(), csr_b.getArrayValues());
    }

    auto operator()(lazy::minus_tag, Vector const& a, UniqueArray<Real> const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit a-b" << std::endl;
#endif
      SimpleCSRVector<Real> const& csr_a = a.impl()->get<BackEnd::tag::simplecsr>();
      return vectorMinus(csr_a.getArrayValues(), b);
    }

    auto operator()(lazy::dot_tag, const VectorDistribution* distribution,
                    Vector const& a, Vector const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit dot(a,b)" << std::endl;
#endif
      return vectorScalProduct(a, b);
    }

    auto operator()(lazy::dot_tag, VectorDistribution const* distribution,
                    Vector const& a, UniqueArray<Real> const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit dot(a,b)" << std::endl;
#endif
      return vectorScalProduct(distribution, a, b);
    }

    auto operator()(lazy::dot_tag, VectorDistribution const* distribution,
                    UniqueArray<Real> const& a, Vector const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit dot(a,b)" << std::endl;
#endif
      return vectorScalProduct(distribution, a, b);
    }

    auto operator()(lazy::dot_tag, VectorDistribution const* distribution,
                    UniqueArray<Real> const& a, UniqueArray<Real> const& b)
    {
#ifdef DEBUG
      std::cout << "\t visit dot(a,b)" << std::endl;
#endif
      return vectorScalProduct(distribution, a, b);
    }

    template <class A, class B>
    auto operator()(lazy::minus_tag, A a, B b)
    {
      return a - b;
    }
  };

  template <typename Tag>
  struct kernel_evaluator
  {

    template <typename T>
    auto operator()(lazy::cst_tag, T c) { return c; }

    template <typename T>
    T const& operator()(lazy::ref_tag, T const& r) { return r; }

    template <typename T>
    auto operator()(lazy::mult_tag, Matrix const& a, UniqueArray<T> const& b)
    {
      return matrixMultT<Tag>(a, b);
    }

    auto operator()(lazy::mult_tag, Matrix const& a, Vector const& b)
    {
      auto const& tag_matrix = a.impl()->template get<Tag>();
      auto const& tag_b = b.impl()->template get<Tag>();
      tag_b.resize(tag_matrix.getAllocSize());
      return matrixMultT<Tag>(a, tag_b.getArrayValues());
    }

    auto operator()(lazy::mult_tag, Real lambda, Vector const& b)
    {
      auto const& tag_b = b.impl()->template get<Tag>();
      return vectorMultT<Tag>(lambda, tag_b.getArrayValues());
    }

    auto operator()(lazy::mult_tag, Real lambda, Matrix const& a)
    {
      return matrixScalT<Tag, Real>(lambda, a);
    }

    auto operator()(lazy::add_tag, Vector const& a, Vector const& b)
    {
      auto const& csr_a = a.impl()->template get<Tag>();
      auto const& csr_b = b.impl()->template get<Tag>();
      return vectorAddT<Tag>(csr_a.getArrayValues(), csr_b.getArrayValues());
    }

    auto operator()(lazy::add_tag, Vector const& a, UniqueArray<Real> const& b)
    {
      auto const& csr_a = a.impl()->template get<Tag>();
      return vectorAddT<Tag>(csr_a.getArrayValues(), b);
    }

    template <class T>
    auto operator()(lazy::add_tag, UniqueArray<T> const& a, UniqueArray<T> const& b)
    {
      return vectorAddT<Tag>(a, b);
    }

    auto operator()(lazy::add_tag, Matrix const& a, Matrix const& b)
    {
      return matrixAddT<Tag>(a, b);
    }

    auto operator()(lazy::minus_tag, Vector const& a, Vector const& b)
    {
      auto const& csr_a = a.impl()->template get<Tag>();
      auto const& csr_b = b.impl()->template get<Tag>();
      return vectorMinusT<Tag>(csr_a.getArrayValues(), csr_b.getArrayValues());
    }

    auto operator()(lazy::minus_tag, Vector const& a, UniqueArray<Real> const& b)
    {
      auto const& csr_a = a.impl()->template get<Tag>();
      return vectorMinusT<Tag>(csr_a.getArrayValues(), b);
    }

    auto operator()(lazy::dot_tag, const VectorDistribution* distribution,
                    Vector const& a, Vector const& b)
    {
      return vectorScalProductT<Tag>(a, b);
    }

    auto operator()(lazy::dot_tag, VectorDistribution const* distribution,
                    Vector const& a, UniqueArray<Real> const& b)
    {
      return vectorScalProductT<Tag>(distribution, a, b);
    }

    auto operator()(lazy::dot_tag, VectorDistribution const* distribution,
                    UniqueArray<Real> const& a, Vector const& b)
    {
      return vectorScalProductT<Tag>(distribution, a, b);
    }

    auto operator()(lazy::dot_tag, VectorDistribution const* distribution,
                    UniqueArray<Real> const& a, UniqueArray<Real> const& b)
    {
      return vectorScalProductT<Tag>(distribution, a, b);
    }

    template <class A, class B>
    auto operator()(lazy::minus_tag, A a, B b)
    {
      return a - b;
    }
  };

  struct size_evaluator
  {
    template <class T>
    auto operator()(lazy::cst_tag, T c)
    {
      return std::numeric_limits<size_t>::max();
    }

    auto operator()(lazy::ref_tag, Matrix const& r) { return r.rowSpace().size(); }

    template <class T>
    auto operator()(lazy::ref_tag, T const& r)
    {
      return r.rowSpace().size();
    }

    template <class T, class A, class B>
    auto operator()(T, A a, B b)
    {
      return std::min(a, b);
    }
  };

  std::size_t allocSize(Matrix const& A)
  {
    SimpleCSRMatrix<Real> const& csr_matrix = A.impl()->get<BackEnd::tag::simplecsr>();
    return csr_matrix.getLocalSize() + csr_matrix.getGhostSize();
  }

  std::size_t allocSize(Vector const& x)
  {
    SimpleCSRVector<Real> const& csr_x = x.impl()->get<BackEnd::tag::simplecsr>();
    return csr_x.getAllocSize();
  }

  struct allocsize_evaluator
  {
    template <class T>
    auto operator()(lazy::cst_tag, T c) { return std::size_t(0); }

    auto operator()(lazy::ref_tag, Matrix const& r) { return allocSize(r); }

    auto operator()(lazy::ref_tag, Vector const& r) { return allocSize(r); }

    template <typename L>
    auto operator()(lazy::add_tag, L const& l, Vector const& r)
    {
      return allocSize(r);
    }

    template <typename L>
    auto operator()(lazy::minus_tag, L const& l, Vector const& r)
    {
      return allocSize(r);
    }

    template <typename L>
    auto operator()(lazy::mult_tag, L const& l, Vector const& r)
    {
      return allocSize(r);
    }
  };

  struct distribution_evaluator
  {
    template <class T>
    VectorDistribution const* operator()(lazy::cst_tag, T c)
    {
      return nullptr;
    }

    VectorDistribution const* operator()(lazy::ref_tag, Matrix const& r)
    {
      return &r.distribution().rowDistribution();
    }

    VectorDistribution const* operator()(lazy::ref_tag, Vector const& r)
    {
      return &r.distribution();
    }

    template <typename L>
    VectorDistribution const* operator()(lazy::add_tag, L const& l, Vector const& r)
    {
      return &r.distribution();
    }

    template <typename L>
    auto operator()(lazy::minus_tag, L const& l, Vector const& r)
    {
      return &r.distribution();
    }

    auto operator()(
    lazy::mult_tag, VectorDistribution const* l, VectorDistribution const* r)
    {
      if (l)
        return l;
      else
        return r;
    }

    template <typename R>
    auto operator()(lazy::mult_tag, Matrix const& l, Vector const& r)
    {
      return &r.distribution();
    }

    template <typename R>
    auto operator()(lazy::mult_tag, Matrix const& l, R const& r)
    {
      return &l.distribution().rowDistribution();
    }

    template <typename L>
    auto operator()(lazy::mult_tag, L const& l, Vector const& r)
    {
      return &r.distribution();
    }
  };

  template <class E>
  auto base_eval(E const& expr) { return expr(cpu_evaluator()); }
  template <class E>
  auto eval(E const& expr) { return expr(cpu_evaluator()); }

  auto operator*(Matrix const& l, Vector const& r) { return mul(ref(l), ref(r)); }

  auto operator*(Real lambda, Vector const& r)
  {
#ifdef DEBUG
    std::cout << "lambda*x" << std::endl;
#endif
    return mul(cst(lambda), ref(r));
  }

  auto operator*(Real lambda, Matrix const& r)
  {
#ifdef DEBUG
    std::cout << "lambda*A" << std::endl;
#endif
    return mul(cst(lambda), ref(r));
  }

  template <typename R>
  auto operator*(Matrix const& l, R const& r)
  {
    return mul(ref(l), r);
  }

  /*
  template<typename L, typename R>
  auto operator*(L&& l, R&& r)
  {
    return mul(l,r) ;
  }*/

  auto operator+(Vector const& l, Vector const& r) { return add(ref(l), ref(r)); }

  template <typename R>
  auto operator+(Vector const& l, R const& r)
  {
    return add(ref(l), r);
  }

  template <typename L>
  auto operator+(L const& l, Vector const& r)
  {
    return add(l, ref(r));
  }

  auto operator+(Matrix const& l, Matrix const& r) { return add(ref(l), ref(r)); }

  /*
  template<typename L, typename R>
  auto operator+(L&& l, R&& r)
  {
    return add(l,r) ;
  }*/

  template <typename R>
  auto operator-(Vector& l, R&& r) { return minus(ref(l), r); }

  template <typename L>
  auto operator-(L&& l, Vector& r) { return minus(l, ref(r)); }

  template <typename L, typename R>
  auto operator-(L&& l, R&& r) { return minus(l, r); }

  auto dot(Vector const& x, Vector const& y) { return scalMul(ref(x), ref(y)); }

  template <typename R>
  auto dot(Vector const& x, R const& y)
  {
    return scalMul(ref(x), y);
  }

  template <typename L>
  auto dot(L const& x, Vector const& y)
  {
    return scalMul(x, ref(y));
  }

  template <typename L, typename R>
  auto dot(L const& x, R const& y)
  {
    return scalMul(x, y);
  }

  template <class E>
  void assign(Vector& y, E const& expr)
  {
    SimpleCSRVector<Real>& csr_y = y.impl()->get<BackEnd::tag::simplecsr>(true);
    csr_y.allocate();
    csr_y.setArrayValues(expr(cpu_evaluator()));
  }

  template <typename Tag, class E>
  void kassign(Vector& y, E const& expr)
  {
    auto& backend_y = y.impl()->template get<Tag>(true);
    backend_y.allocate();
    backend_y.setArrayValues(expr(kernel_evaluator<Tag>()));
  }

  template <typename A, typename B>
  auto vassign(A&& a, B&& b)
  {
    return [&](auto visitor) { return visitor(lazy::assign_tag{}, a, b); };
  }

  template <typename E>
  auto veval(double& value, E&& e)
  {
    return [&](auto visitor) { return visitor(lazy::eval_tag{}, value, e); };
  }

  struct pipeline_evaluator
  {
    template <typename T>
    void eval(T&& expr)
    {
#ifdef DEBUG
      std::cout << "pipeline eval" << std::endl;
#endif
      expr(*this);
    }
    template <typename T0, typename... T>
    void eval(T0&& expr0, T&&... args)
    {
      eval(expr0);
      eval(args...);
    }

    template <typename E>
    auto operator()(lazy::assign_tag, Vector& a, E&& expr)
    {
#ifdef DEBUG
      std::cout << "vector assignement" << std::endl;
#endif
      assign(a, expr);
    }

    template <typename E>
    auto operator()(lazy::eval_tag, double& value, E&& expr)
    {
      value = base_eval(expr);
      return value;
    }
  };

  template <typename... T>
  void pipeline(T... args)
  {
    pipeline_evaluator evaluator;
    evaluator.eval(args...);
  }
} // namespace MVExpr

template <typename E>
Vector&
Vector::operator=(E const& expr)
{
  SimpleCSRVector<Real>& csr_y = impl()->get<BackEnd::tag::simplecsr>(true);
  csr_y.allocate();
  csr_y.setArrayValues(expr(MVExpr::cpu_evaluator()));
  return *this;
}

template <typename E>
Matrix&
Matrix::operator=(E const& expr)
{
  *this = expr(MVExpr::cpu_evaluator());
  return *this;
}

} // namespace Alien

#endif /* ALIEN_EXPRESSION_MVEXPR_MVEXPR_H_ */
