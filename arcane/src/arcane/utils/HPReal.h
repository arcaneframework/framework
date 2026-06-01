// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HPReal.h                                                    (C) 2000-2019 */
/*                                                                           */
/* High-precision real number.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HPREAL_H
#define ARCANE_UTILS_HPREAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Numeric.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class implementing a High-Precision real number.
 
 This class is based on the article:

 Obtaining identical results with double precision global accuracy on
 different numbers of processors in parallel particle Monte Carlo
 simulations (Mathew A. Cleveland, Thomas A. Brunner, Nicholas A. Gentile,
 jeffrey A. Keasler) in Journal Of Computational Physics 251 (2013) 223-236.

 Possible operations are accumulation (operator+=()),
 reduction (reduce()) and conversion to a Real (toReal()).
 Other operations such as division or multiplication are not possible.

 To conform to the classic 'Real' type, the default constructor
 of this class performs no initialization.

 The toReal() method allows converting the HPReal into a classic Real.
 The typical usage is as follows:

 \code
 * HPReal r(0.0);
 * ArrayView<Real> x = ...;
 * for( Integer i=0, n=x.size(); i<n; ++i ){
 *   // Accumulates the value
 *   r += x[i];
 * }
 * Real final_r = r.toReal();
 \endcode

 Evolution 12/18/18:

 Correction of the base algorithm error for static HPReal _doTwoSum(Real a, Real b)

   Real sum_error = (a - (value-approx_b) + (b-approx_b));

 became (in accordance with Ref 1)

   Real sum_error = (a - (value-approx_b)) + (b-approx_b);

 Correction of the error in static HPReal _doQuickTwoSum(Real a,Real b)

   it is necessary to test if a>b and invert a and b if this is not the case (see Ref 2, 3, 4)

 Modification of Accumulate and Reduce,
 Accumulate and Reduce were based on Ref 1, which is based on Ref 4

 Addition of products as proposed in Ref 2 (in accordance with Ref 5)
 of HPReal*Real or HPReal*HPReal (we take the same value of p for the
 SPLIT() function as in Ref 2, 4, 5)

 Note that the Real*Real product in Ref 4 (p.4) is not written in the same way as those in Ref 2 (p.2) or 5 (p.3)

 +++++++++++++++++

 Overall, reduction algorithms or errors accumulated only with addition are less good

 Algorithm 6 of DD_TWOSUM [Ref 2] is also less good:

 \code
 static HPReal accumulate(Real a,MYHPReal b)
 {
   HPReal x(_doTwoSum(a,b.value()));
   Real d= x.correction() + b.correction();
   HPReal u(_doQuickTwoSum(x.value(),d));
   return _doQuickTwoSum(u.value(),u.correction());
 }
 \endcode

 or

 \code
 HPReal  reduce(HPReal a,HPReal b)
 {
   HPReal x(_doTwoSum(a.value(),b.value()));
   HPReal y(_doTwoSum(a.correction(),b.correction()));
   Real d= x.correction() + y.value();
   HPReal u(_doQuickTwoSum(x.value(),d));
   Real w=y.correction()+u.correction();
   return _doQuickTwoSum(u.value(),w);
 }
 \endcode

 Tests on millions of positive as well as negative values have been performed
 for addition, multiplication, and division in different orders.

 I have not managed to show a difference in results in a simple case between the proposed algorithm for the sum and the one initially coded.

 References
 ----------

 Ref 1
  Obtaining identical results with double precision global accuracy on
  different numbers of processors in parallel particle Monte Carlo
  simulations (Mathew A. Cleveland, Thomas A. Brunner, Nicholas A. Gentile,
  jeffrey A. Keasler) in Journal Of Computational Physics 251 (2013) 223-236.

 Ref 2
   Automatic Source-to-Source error compensation of floating-point programs
   L.Thevenoux, Ph langlois, Mathieu Martel
   HAL Id: hal-01158399

 Ref 3
   Numerical validation of compensated summation algorithms withs stochastic arithmetic
   S.Gaillat, F.Jézéquel , R.Picot
   Published in Electronic Notes in Theoritical Computer Science

   or

   Numerical validation of compensated algorithms withs stochastic arithmetic
   S.Gaillat, F.Jézéquel , R.Picot
   Published in Applied Mathematics and Computation 329 (2018)339-363

 Ref 4
   Library for Double-Double and Quad-Double Arithmetic
   December 29, 2007
   Yozo Hida Xiaoye Li D.H.Bailey

 Ref 5
   Accurate floating point Product and exponentation
   S.Graillat
   HAL ID: hal-00164607

 Ref 6
   A floating point technique for extending the avaible precision
   T.J.Dekker
   Numeri.Math 18, 224-242 (1971)
*/
class ARCANE_UTILS_EXPORT HPReal
{
 public:
  /*!
   * \brief Default constructor without initialization.
   */
  HPReal() {}

  //! Creates an HP real with the value \a value and the correction \a correction
  explicit HPReal(double avalue)
  : m_value(avalue)
  , m_correction(0.0)
  {}

  //! Creates an HP real with the value \a value and the correction \a correction
  HPReal(double avalue, double acorrection)
  : m_value(avalue)
  , m_correction(acorrection)
  {}

  //! Internal value. Generally, you must use toReal()
  Real value() const { return m_value; }

  //! Internal correction.
  Real correction() const { return m_correction; }

  //! Adds a Real while preserving the error.
  void operator+=(Real v)
  {
    *this = accumulate(v, *this);
  }

  //! Adds an HPReal \a v while preserving the error (reduction)
  inline void operator+=(HPReal v)
  {
    *this = reduce(*this, v);
  }

  //! Converts the instance to a Real.
  inline Real toReal() const
  {
    return value() + correction();
  }

  //! Adds an HPReal \a v while preserving the error (reduction)
  HPReal reduce(HPReal b) const
  {
    return reduce(*this, b);
  }

  //! Multiplies a Real while preserving the error.
  void operator*=(Real v)
  {
    *this = product(v, *this);
  }

  //! Multiplies an HPReal \a v while preserving the error (reduction)
  inline void operator*=(HPReal v)
  {
    *this = product(*this, v);
  }

  //! Multiplies a Real while preserving the error.
  void operator/=(Real v)
  {
    *this = div2(v, *this);
  }

  //! Multiplies an HPReal \a v while preserving the error (reduction)
  inline void operator/=(HPReal v)
  {
    *this = div2(*this, v);
  }

  /*!
   * \brief Reads an HPReal from the stream \a i.
   * The pair is read in the form of two #Real type values.
   */
  std::istream& assign(std::istream& i);
  //! Writes the instance to the stream \a o readable by an assign()
  std::ostream& print(std::ostream& o) const;
  //! Writes the instance to the stream \a o in the form (x,y)
  std::ostream& printPretty(std::ostream& o) const;

 public:
  //! Zero value.
  static HPReal zero() { return HPReal(0.0); }

 public:
  static HPReal accumulate(Real a, HPReal b)
  {
    HPReal x(_doTwoSum(a, b.value()));
    Real c = x.correction() + b.correction();
    return _doQuickTwoSum(x.value(), c);
  }

  // Mpi passes through
  static HPReal reduce(HPReal a, HPReal b)
  {
    HPReal x(_doTwoSum(a.value(), b.value()));
    Real c = x.correction() + a.correction() + b.correction();
    return _doQuickTwoSum(x.value(), c);
  }

  // algo 11 of AC_TWOPRODUCTS Ref 2
  static HPReal product(HPReal a, HPReal b)
  {
    HPReal x(_doTwoProducts(a.value(), b.value()));
    Real w = x.correction() + (a.value() * b.correction() + b.value() * a.correction());
    return HPReal(x.value(), w);
  }
  // algo 11 of AC_TWOPRODUCTS Ref 2
  static HPReal product(Real a, HPReal b)
  {
    HPReal x(_doTwoProducts(a, b.value()));
    Real w = x.correction() + (a * b.correction());
    return HPReal(x.value(), w);
  }

  // algo div2 Ref 6    div2 =  a/b
  // I put parentheses in the evaluation of w to force an order sometimes the compiler options
  // ATTENTION the last 2 lines of the div2 algorithm are not included (we do not renormalize)
  // The "mul2" algorithm also contains them and did not include them for "product" because algorithm 11 of Ref 2 does not have them (in my humble opinion)
  static HPReal div2(HPReal a, HPReal b)
  {
    Real c = a.value() / b.value();
    HPReal u(_doTwoProducts(c, b.value()));
    Real w = ((((a.value() - u.value()) + -u.correction()) + a.correction()) - c * b.correction()) / b.value();
    return HPReal(c, w);
  }
  // algo div2 Ref 6     div2 =  a/b
  static HPReal div2(Real b, HPReal a)
  {
    Real c = a.value() / b;
    HPReal u(_doTwoProducts(c, b));
    Real w = (((a.value() - u.value()) + -u.correction()) + a.correction()) / b;
    return HPReal(c, w);
  }

 private:
  Real m_value;
  Real m_correction;

 private:
  // Correction of ()
  static HPReal _doTwoSum(Real a, Real b)
  {
    Real value = a + b;
    Real approx_b = value - a;
    Real sum_error = (a - (value - approx_b)) + (b - approx_b);
    return HPReal(value, sum_error);
  }
  // correction tests of the values of a and b and we put absolute values
  static HPReal _doQuickTwoSum(Real a1, Real b1)
  {
    Real a = a1;
    Real b = b1;
    if (std::abs(b1) > std::abs(a1)) {
      a = b1;
      b = a1;
    }
    Real value = a + b;
    Real error_value = (b - (value - a));
    return HPReal(value, error_value);
  }
  // algorithm 4 ref 2 or algorithm 2.4 ref 3 or algorithm 6 p.4
  static HPReal _doTwoProducts(Real a, Real b)
  {
    Real x = a * b;
    HPReal aw = SPLIT(a);
    HPReal bw = SPLIT(b);
    Real y = aw.correction() * bw.correction() - (((x - aw.value() * bw.value()) - aw.correction() * bw.value()) - aw.value() * bw.correction());
    return HPReal(x, y);
  }
  static HPReal SPLIT(Real a)
  {
    const Real f = 134217729; // 1+ 2^ceil(52/2);
    Real c = f * a;
    Real x = c - (c - a);
    Real y = a - x;
    return HPReal(x, y);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator<(const HPReal& a, const HPReal& b)
{
  return a.toReal() < b.toReal();
}

inline bool
operator>(const HPReal& a, const HPReal& b)
{
  return a.toReal() > b.toReal();
}

inline bool
operator==(const HPReal& a, const HPReal& b)
{
  return a.value() == b.value() && a.correction() == b.correction();
}

inline bool
operator!=(const HPReal& a, const HPReal& b)
{
  return !operator==(a, b);
}

inline HPReal
operator+(const HPReal& a, const HPReal& b)
{
  return HPReal::reduce(a, b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o, HPReal t)
{
  return t.printPretty(o);
}

inline std::istream&
operator>>(std::istream& i, HPReal& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
