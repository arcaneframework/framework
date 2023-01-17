// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Matrix.cc                                                   (C) 2000-2022 */
/*                                                                           */
/* Matrix d'algèbre linéraire.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/Numeric.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/utils/ValueConvert.h"

#include "arcane/matvec/Matrix.h"
#include "arcane/matvec/Vector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MatVec
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Matrice avec stockage CSR.
 */
class MatrixImpl
{
 public:
  MatrixImpl() : m_nb_reference(0), m_nb_row(0), m_nb_column(0), m_nb_element(0) {}
  MatrixImpl(Integer nb_row,Integer nb_column);
 private:
  void operator=(const Matrix& rhs);
 public:
  MatrixImpl* clone()
  {
    MatrixImpl* m2 = new MatrixImpl();
    m2->m_nb_row = m_nb_row;
    m2->m_nb_column = m_nb_column;
    m2->m_nb_element = m_nb_element;
    m2->m_values = m_values.clone();
    m2->m_rows_index = m_rows_index.clone();
    m2->m_columns = m_columns.clone();
    return m2;
  }
 private:
 public:
  Integer nbRow() const { return m_nb_row; }
  Integer nbColumn() const { return m_nb_column; }
  void setRowsSize(IntegerConstArrayView rows_size);
  void setValues(IntegerConstArrayView columns,RealConstArrayView values);
  void dump(std::ostream& o);
  RealConstArrayView values() const { return m_values; }
  RealArrayView values() { return m_values; }
  Real value(Integer row,Integer column) const;
  IntegerConstArrayView rowsIndex() const { return m_rows_index; }
  IntegerConstArrayView columns() const { return m_columns; }
  IntegerArrayView rowsIndex() { return m_rows_index; }
  IntegerArrayView columns() { return m_columns; }
  void setValue(Integer row,Integer column,Real value);
  void sortDiagonale();
  void assemble();
 public:
  void checkValid();
 public:
  void addReference()
  {
    ++m_nb_reference;
  }
  void removeReference()
  {
    --m_nb_reference;
    if (m_nb_reference==0)
      delete this;
  }
 private:
  Integer m_nb_reference;
  Integer m_nb_row;
  Integer m_nb_column;
  Integer m_nb_element;
  SharedArray<Real> m_values;
  SharedArray<Integer> m_rows_index;
  SharedArray<Integer> m_columns;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixImpl::
MatrixImpl(Integer nb_row,Integer nb_column)
: m_nb_reference(0)
, m_nb_row(nb_row)
, m_nb_column(nb_column)
, m_nb_element(0)
, m_rows_index(m_nb_row+1)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix::
Matrix(Integer nb_row,Integer nb_column)
: m_impl(new MatrixImpl(nb_row,nb_column))
{
  m_impl->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix::
Matrix(MatrixImpl* impl)
: m_impl(impl)
{
  if (m_impl)
    m_impl->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix::
Matrix(const Matrix& rhs)
: m_impl(rhs.m_impl)
{
  if (m_impl)
    m_impl->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Matrix::
operator=(const Matrix& rhs)
{
  if (rhs.m_impl)
    rhs.m_impl->addReference();
  if (m_impl)
    m_impl->removeReference();
  m_impl = rhs.m_impl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix::
~Matrix()
{
  if (m_impl)
    m_impl->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Matrix::
nbRow() const
{
  return m_impl->nbRow();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Matrix::
sortDiagonale()
{
  return m_impl->sortDiagonale();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Matrix::
nbColumn() const
{
  return m_impl->nbColumn();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix Matrix::
clone() const
{
  MatrixImpl* m = m_impl->clone();
  return Matrix(m);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IntegerConstArrayView Matrix::
rowsIndex() const
{
  return m_impl->rowsIndex();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Matrix::
dump(std::ostream& o) const
{
  m_impl->dump(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Matrix::
setValue(Integer row,Integer column,Real value)
{
  m_impl->setValue(row,column,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real Matrix::
value(Integer row,Integer column) const
{
  return m_impl->value(row,column);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Matrix::
setValues(IntegerConstArrayView columns,RealConstArrayView values)
{
  m_impl->setValues(columns,values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Matrix::
setRowsSize(IntegerConstArrayView rows_size)
{
  m_impl->setRowsSize(rows_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealConstArrayView Matrix::
values() const
{
  return m_impl->values();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealArrayView Matrix::
values()
{
  return m_impl->values();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IntegerArrayView Matrix::
columns()
{
  return m_impl->columns();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IntegerArrayView Matrix::
rowsIndex()
{
  return m_impl->rowsIndex();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IntegerConstArrayView Matrix::
columns() const
{
  return m_impl->columns();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DiagonalPreconditioner::
DiagonalPreconditioner(const Matrix& matrix)
: m_inverse_diagonal(matrix.nbRow())
{
  Integer size = m_inverse_diagonal.size();
  IntegerConstArrayView rows_index = matrix.rowsIndex();
  IntegerConstArrayView columns = matrix.columns();
  RealConstArrayView mat_values = matrix.values();
  RealArrayView vec_values = m_inverse_diagonal.values();
  for( Integer i=0, is=size; i<is; ++i ){
    for( Integer z=rows_index[i] ,zs=rows_index[i+1]; z<zs; ++z ){
      Integer mj = columns[z];
      if (mj==i)
        vec_values[i] = mat_values[z];
    }
  }
#if 0
  cout << "DIAG1=";
  m_inverse_diagonal.dump(cout);
  cout << "\n";
#endif
  const double epsilon = std::numeric_limits<Real>::min();

  // Inverse la diagonale si la valeur n'est pas inférieure à l'epsilon de Real
  // (sinon cela génère un FPE)
  for( Integer i=0; i<size; ++i ){
    Real v = vec_values[i];
    bool is_zero  = v>-epsilon && v<epsilon;
    vec_values[i] = (is_zero) ? 1.0 : (1.0 / v);
  }

#if 0
  cout << "DIAG2=";
  m_inverse_diagonal.dump(cout);
  cout << "\n";
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DiagonalPreconditioner::
apply(Vector& out_vec,const Vector& vec)
{
  Integer size = m_inverse_diagonal.size();
  RealConstArrayView inverse_diagonal_values = m_inverse_diagonal.values();
  RealConstArrayView vec_values = vec.values();
  RealArrayView out_vec_values = out_vec.values();
  for( Integer i=0; i<size; ++i )
    out_vec_values[i] = vec_values[i] * inverse_diagonal_values[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixOperation::
matrixVectorProduct(const Matrix& mat,const Vector& vec,Vector& out_vec)
{
  Integer nb_row = mat.nbRow();
  Integer nb_column = mat.nbColumn();
  //cout << "** MATRIX ROW=" << nb_row << " COL=" << nb_column
  //     << " intput_size=" << vec.size()
  //     << " output_size=" << out_vec.size()
  //     << "\n";
  if (nb_column!=vec.size())
    throw ArgumentException("MatrixVectorProduct","Bad size for input vector");
  if (nb_row!=out_vec.size())
    throw ArgumentException("MatrixVectorProduct","Bad size for output_vector");
  IntegerConstArrayView rows_index = mat.rowsIndex();
  IntegerConstArrayView columns = mat.columns();
  RealConstArrayView mat_values = mat.values();
  RealConstArrayView vec_values = vec.values();
  RealArrayView out_vec_values = out_vec.values();
    
  for( Integer i=0, is=nb_row; i<is; ++i ){
    Real sum = 0.0;
    for( Integer z=rows_index[i] ,zs=rows_index[i+1]; z<zs; ++z ){
      Integer mj = columns[z];
      Real mv = mat_values[z];
      //cout << "ADD vec=" << vec_values[mj] << " mv=" << mv << " sum=" << sum << '\n';
      sum += vec_values[mj]*mv;
    }
    out_vec_values[i] = sum;
  }
}
 
Real MatrixOperation::
dot(const Vector& vec)
{
  Integer size = vec.size();
  RealConstArrayView vec_values = vec.values();
  Real v = 0.0;
  for( Integer i=0; i<size; ++i )
    v += vec_values[i] * vec_values[i];
  return v;
}

Real MatrixOperation::
dot(const Vector& vec1,const Vector& vec2)
{
  Real v = 0.0;
  Integer size = vec1.size();
  RealConstArrayView vec1_values = vec1.values();
  RealConstArrayView vec2_values = vec2.values();
  for( Integer i=0; i<size; ++i ){
    v += vec1_values[i] * vec2_values[i];
    //cout << " i=" << i << " v=" << v << '\n';
  }
  return v;
}

void MatrixOperation::
negateVector(Vector& vec)
{
  Integer size = vec.size();
  RealArrayView vec_values = vec.values();
  for( Integer i=0; i<size; ++i )
    vec_values[i] = -vec_values[i];
}

void MatrixOperation::
scaleVector(Vector& vec,Real mul)
{
  Integer size = vec.size();
  RealArrayView vec_values = vec.values();
  for( Integer i=0; i<size; ++i )
    vec_values[i] *= mul;
}

void MatrixOperation::
addVector(Vector& out_vec,const Vector& vec)
{
  Integer size = vec.size();
  RealConstArrayView vec_values = vec.values();
  RealArrayView out_vec_values = out_vec.values();
  for( Integer i=0; i<size; ++i )
    out_vec_values[i] += vec_values[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConjugateGradientSolver::
_applySolver(const Matrix& a,const Vector& b,Vector& x,Real epsilon,IPreconditioner* p)
{
  MatrixOperation mat_op;
  Integer vec_size = a.nbRow();
  Vector r(vec_size);

  // r = b - Ax
  mat_op.matrixVectorProduct(a,x,r);
  mat_op.negateVector(r);
  mat_op.addVector(r,b);

  m_nb_iteration = 0;
  m_residual_norm = 0.0;

  /*cout << " R=";
  r.dump(cout);
  cout << '\n';*/

  Vector d(r.size());
  if (p)
    p->apply(d,r);
  else
    d.copy(r);

  Vector q(r.size());
  Vector t(r.size());
  Vector s(r.size());
  Real delta_new = 0.0;
  //Real r0=mat_op.dot(r);
  if (p){
    delta_new = mat_op.dot(r,d);
  }
  else
    delta_new = mat_op.dot(r);
#if 0
  cout << "R=";
  r.dump(cout);
  cout << "\n";
  cout << "D=";
  d.dump(cout);
  cout << "\n";
#endif
  //Real norm0 = r.normInf();
  Real delta0 = delta_new;
  //cout << " TOL=" << epsilon << " delta0=" << delta0 << '\n';
  //cout << " deltanew=" << delta_new << '\n';
  Integer nb_iter = 0;
  for( nb_iter=0; nb_iter<m_max_iteration; ++nb_iter ){
    if (delta_new < epsilon*epsilon*delta0)
      break;
#if 0
    // Si on utilise la norme inf
    {
      Real norm = r.normInf();
      cout << " norm=" << norm << " norm0=" << norm0 << '\n';
      if (norm < epsilon * norm0)
        break;
    }
#endif

    //cout << "delta_new=" << delta_new << '\n';
    // q = A * d
    mat_op.matrixVectorProduct(a,d,q);
    Real alpha = delta_new / mat_op.dot(d,q);

    // x = x + alpha * d
    t.copy(d);
    mat_op.scaleVector(t,alpha);
    mat_op.addVector(x,t);

#if 0
    // r = b - Ax
    mat_op.matrixVectorProduct(a,x,r);
    mat_op.negateVector(r);
    mat_op.addVector(r,b);
#endif
    // r = r - alpha * q
    mat_op.scaleVector(q,-alpha);
    mat_op.addVector(r,q);

    if (p)
      p->apply(s,r);
    Real delta_old = delta_new;
    if (p)
      delta_new = mat_op.dot(r,s);
    else
      delta_new = mat_op.dot(r);
    Real beta = delta_new / delta_old;
    //cout << " alpha=" << alpha << " beta=" << beta << " delta_new=" << delta_new << '\n';
    // d = beta * d + s
    mat_op.scaleVector(d,beta);
    if (p){
      //mat_op.addVector(s,r);
      mat_op.addVector(d,s);
    }
    else
      mat_op.addVector(d,r);
    //cout << '\n';
  }
  //cout << " X=";
  //x.dump(cout);
  //cout << '\n';
  //cout << "NB ITER=" << nb_iter << " epsilon=" << epsilon
  //     << " delta0=" << delta0
  //     << " delta_new=" << delta_new << " r=" << mat_op.dot(r) << " r0=" << r0 << '\n';
  m_nb_iteration = nb_iter;
  m_residual_norm = delta_new;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConjugateGradientSolver::
_applySolverAsHypre(const Matrix& a,const Vector& b,Vector& x,Real tol,
                    IPreconditioner* precond)
{
  //cout << "** APPLY ARCANE_SOLVER PCG using hypre method\n";
  ARCANE_CHECK_POINTER(precond);

  MatrixOperation mat_op;
  Integer vec_size = a.nbRow();
  Vector r(vec_size);
  Vector p(vec_size);

  const bool is_two_norm = true;
  Real bi_prod = 0.0;
  if (is_two_norm){
    bi_prod = mat_op.dot(b,b);
  }
  else{
    precond->apply(p,b);
    bi_prod = mat_op.dot(p,b);
  }
  Real eps = tol*tol;
  //TODO: regarder stop_crit
    
  // r = b - Ax
  mat_op.matrixVectorProduct(a,x,r);
  mat_op.negateVector(r);
  mat_op.addVector(r,b);

  m_nb_iteration = 0;
  m_residual_norm = 0.0;

  /*cout << " R=";
  r.dump(cout);
  cout << '\n';*/

  Vector tmp_x(x.size());

  Vector d(r.size());
  Vector s(r.size());
  d.copy(r);
  // p = C*r
  precond->apply(p,r);

  /* gamma = <r,p> */
  Real gamma = mat_op.dot(r,p);

  Real i_prod = 0.0;

  //Real delta_new = 0.0;
  //Real r0=mat_op.dot(r);
  //if (p){
  //  delta_new = mat_op.dot(r,d);
  //}
  //else
  // delta_new = mat_op.dot(r);
#if 0
  cout << "R=";
  r.dump(cout);
  cout << "\n";
  cout << "D=";
  d.dump(cout);
  cout << "\n";
#endif
  //Real delta0 = delta_new;
  //cout << "** TOL=" << tol << " bi_prod=" << bi_prod << '\n';
  //cout << " deltanew=" << delta_new << '\n';
  Integer nb_iter = 0;
  for( nb_iter=0; nb_iter<m_max_iteration; ++nb_iter ){

    // s = A * p
    mat_op.matrixVectorProduct(a,p,s);

    /* alpha = gamma / <s,p> */
    Real sdotp = mat_op.dot(s, p);

    Real alpha = gamma / sdotp;

    Real gamma_old = gamma;

    /* x = x + alpha*p */
    mat_op.matrixVectorProduct(a,p,tmp_x);
    mat_op.addVector(x,tmp_x);

    /* r = r - alpha*s */
    tmp_x.copy(s);
    mat_op.scaleVector(tmp_x,-alpha);
    mat_op.addVector(r,tmp_x);
         
    /* s = C*r */
    precond->apply(s,r);

    /* gamma = <r,s> */
    gamma = mat_op.dot(r,s);

    /* set i_prod for convergence test */
    if (is_two_norm)
      i_prod = mat_op.dot(r,r);
    else
      i_prod = gamma;

    //cout << "** sdotp=" << sdotp << " i_prod=" << i_prod << " bi_prod=" << bi_prod << '\n';

    /* check for convergence */
    if (i_prod / bi_prod < eps){
#if 0
      if (rel_change && i_prod > guard_zero_residual)
      {
        pi_prod = (*(pcg_functions->InnerProd))(p,p); 
        xi_prod = (*(pcg_functions->InnerProd))(x,x);
        ratio = alpha*alpha*pi_prod/xi_prod;
        if (ratio < eps)
        {
          (pcg_data -> converged) = 1;
          break;
        }
      }
      else
      {
#endif
        //(pcg_data -> converged) = 1;
        break;
#if 0
      }
#endif
    }

    /* beta = gamma / gamma_old */
    Real beta = gamma / gamma_old;

    //cout << " alpha=" << alpha << " beta=" << beta << " delta_new=" << gamma << '\n';

    /* p = s + beta p */
    tmp_x.copy(p);
    mat_op.scaleVector(tmp_x,beta);
    mat_op.addVector(tmp_x,s);
    p.copy(tmp_x);
  }
  //cout << " X=";
  //x.dump(cout);
  //cout << '\n';
  //cout << "NB ITER=" << nb_iter << " epsilon=" << epsilon
  //     << " delta0=" << delta0
  //     << " delta_new=" << delta_new << " r=" << mat_op.dot(r) << " r0=" << r0 << '\n';
  m_nb_iteration = nb_iter;
  m_residual_norm = gamma;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ConjugateGradientSolver::
solve(const Matrix& a,const Vector& b,Vector& x,Real epsilon,IPreconditioner* p)
{
  //epsilon = 1e-12;
  //x.values().fill(0.0);
  m_nb_iteration = 0;
  m_residual_norm = 0.0;
  //_doConjugateGradient(a,b,x,epsilon,&p);
  cout.precision(20);
  const bool do_print = false;
  if (do_print){
    cout << "A=";
    a.dump(cout);
    cout << '\n';
    cout << "b=";
    b.dump(cout);
    cout << '\n';
  }

  DiagonalPreconditioner p2(a);
  if (!p)
    p = &p2;
  _applySolver(a,b,x,epsilon,p);

  if (do_print){
    cout << "\n\nSOLUTION\nx=";
    x.dump(cout);
    cout << '\n';
  }

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static void
_testArcaneMatrix1()
{
  Integer s = 5;
  Matrix m(s,s);
  Matrix m1(Matrix::read("test.mat"));
  cout << "** o=" << m.nbRow() << '\n';
  cout << "** M1=";
  m1.dump(cout);
  cout << '\n';
  Vector v1(10);
  for( Integer i=0; i<10; ++i ){
    v1.values()[i] = (Real)(i+1);
  }
  Real epsilon = 1.0e-10;
  {
    Vector v2(10);
    MatrixOperation mat_op;
    Vector r(5);
    mat_op.matrixVectorProduct(m1,v1,v2);
    cout << "** V1=";
    v1.dump(cout);
    cout << "\n";
    cout << "** V2=";
    v2.dump(cout);
    cout << "\n";
    Vector b3(10);
    for( Integer i=0; i<10; ++i ){
      b3.values()[i] = (Real)(i+1);
    }
    Vector x3(10);
    DiagonalPreconditioner p(m1);
    ConjugateGradientSolver solver;
    solver.solve(m1,b3,x3,epsilon,&p);
  }

  IntegerUniqueArray rows_size(s);
  rows_size[0] = 5;
  rows_size[1] = 2;
  rows_size[2] = 2;
  rows_size[3] = 2;
  rows_size[4] = 2;
  m.setRowsSize(rows_size);
  RealUniqueArray values(13);
  IntegerUniqueArray columns(13);
  values[0] = 9.0;
  values[1] = 1.5;
  values[2] = 6.0;
  values[3] = 0.75;
  values[4] = 3.0;

  values[5] = 1.5;
  values[6] = 0.5;

  values[7] = 6.0;
  values[8] = 0.5;

  values[9] = 0.75;
  values[10] = 5.0 / 8.0;

  values[11] = 3.0;
  values[12] = 16.0;

  columns[0] = 0;
  columns[1] = 1;
  columns[2] = 2;
  columns[3] = 3;
  columns[4] = 4;
  columns[5] = 0;
  columns[6] = 1;
  columns[7] = 0;
  columns[8] = 2;
  columns[9] = 0;
  columns[10] = 3;
  columns[11] = 0;
  columns[12] = 4;
  m.setValues(columns,values);
  m.dump(cout);
  cout << '\n';
  Vector b(5);
  RealArrayView rav(b.values());
  rav[0] = 1.0;
  rav[1] = 1.0;
  rav[2] = 1.0;
  rav[3] = 3.0;
  rav[4] = 1.0;
  Vector x(5);
  ConjugateGradientSolver solver;
  solver.solve(m,b,x,epsilon);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _testArcaneMatrix2(Integer matrix_number)
{
  cout.precision(16);
  cout.flags(std::ios::scientific);
  String dir_name = "test-matrix";
  StringBuilder ext_name;
  ext_name += matrix_number;
  ext_name += ".00000";
  Matrix m(Matrix::readHypre(dir_name+"/MATRIX_matrix"+ext_name.toString()));
  Vector b(Vector::readHypre(dir_name+"/MATRIX_b"+ext_name.toString()));
  Vector xref(Vector::readHypre(dir_name+"/MATRIX_x"+ext_name.toString()));
  Vector x(m.nbRow());
  cout << "** XREF=" << xref.size() << '\n';
  //m.dump(cout);
  cout << '\n';
  {
    Real epsilon = 1.0e-15;
    {
      ConjugateGradientSolver solver;
      solver.solve(m,b,x,epsilon);
    }
    MatrixOperation mat_op;
    mat_op.negateVector(xref);
    mat_op.addVector(xref,x);
    //xref.dump(cout);
    Real x_norm = mat_op.dot(x);
    Real diff_norm = mat_op.dot(xref);
    cout << "** MATRIX_NUMBER = " << matrix_number;
    if (!math::isNearlyZero(x_norm)){
      cout << "** norm=" << x_norm << " REL=" << diff_norm/x_norm << '\n';
    }
    else
      cout << "** norm=" << x_norm << " DIFF=" << diff_norm << '\n';
    cout << '\n';
  }
  {
    Real epsilon = 1.0e-15;
    x.values().fill(0.0);
    {
      DiagonalPreconditioner p(m);
      ConjugateGradientSolver solver;
      solver.solve(m,b,x,epsilon,&p);
    }
    MatrixOperation mat_op;
    mat_op.negateVector(xref);
    mat_op.addVector(xref,x);
    //xref.dump(cout);
    Real x_norm = mat_op.dot(x);
    Real diff_norm = mat_op.dot(xref);
    cout << "** PRECOND MATRIX_NUMBER = " << matrix_number;
    if (!math::isNearlyZero(x_norm)){
      cout << "** norm=" << x_norm << " REL=" << diff_norm/x_norm << '\n';
    }
    else
      cout << "** norm=" << x_norm << " DIFF=" << diff_norm << '\n';
    cout << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" void _testArcaneMatrix()
{
  _testArcaneMatrix1();
  //for( Integer i=1; i<15; ++i )
  //_testArcaneMatrix2(i);
  exit(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixImpl::
setRowsSize(IntegerConstArrayView rows_size)
{
  Integer index = 0;
  for( Integer i=0, is=m_nb_row; i<is; ++i ){
    m_rows_index[i] = index;
    index += rows_size[i];
  }
  m_rows_index[m_nb_row] = index;
  m_nb_element = index;
  m_columns.resize(index);
  m_columns.fill(-1);
  m_values.resize(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixImpl::
setValues(IntegerConstArrayView columns,RealConstArrayView values)
{
  m_columns.copy(columns);
  m_values.copy(values);
  if (arcaneIsCheck())
    checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixImpl::
checkValid()
{
  IntegerConstArrayView columns = m_columns;
  IntegerConstArrayView rows_index = m_rows_index;

  Integer nb_column = m_nb_column;
  for( Integer row=0, nb_row = m_nb_row; row<nb_row; ++row ){
    for( Integer j=rows_index[row],js=rows_index[row+1]; j<js; ++j ){
      if (columns[j]>=nb_column || columns[j]<0){
        cout << "BAD COLUMN VALUE for row=" << row << " column=" << columns[j]
             << " column_index=" << j
             << " nb_column=" << nb_column << " nb_row=" << nb_row << '\n';
        throw FatalErrorException("MatrixImpl::checkValid");
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real MatrixImpl::
value(Integer row,Integer column) const
{
  IntegerConstArrayView rows_index = m_rows_index;
  IntegerConstArrayView columns = m_columns;
  RealConstArrayView values = m_values;

  for( Integer z=rows_index[row], zs=rows_index[row+1]; z<zs; ++z ){
    if (columns[z]==column)
      return values[z];
  }
  return 0.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixImpl::
setValue(Integer row,Integer column,Real value)
{
  IntegerConstArrayView rows_index = m_rows_index;
  IntegerArrayView columns = m_columns;
#ifdef ARCANE_CHECK
  if (row>=m_nb_row)
    throw ArgumentException("MatrixImpl::setValue","Invalid row");
  if (column>=m_nb_column)
    throw ArgumentException("MatrixImpl::setValue","Invalid column");
  if (row<0)
    throw ArgumentException("MatrixImpl::setValue","Invalid row");
  if (column<0)
    throw ArgumentException("MatrixImpl::setValue","Invalid column");
#endif
  for( Integer j=rows_index[row],js=rows_index[row+1]; j<js; ++j ){
    if (columns[j]==(-1) || columns[j]==column){
      columns[j] = column;
      m_values[j] = value;
      return;
    }
  }
  throw ArgumentException("Matrix::setValue","column not found");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixImpl::
sortDiagonale()
{
  assemble();
  if (arcaneIsCheck()){
    checkValid();
  }
  // Trie la diagonale pour qu'elle soit le premièr élément de la ligne
  IntegerConstArrayView rows_index = m_rows_index;
  IntegerArrayView columns = m_columns;
  RealArrayView values = m_values;
  for( Integer row=0, nb_row = m_nb_row; row<nb_row; ++row ){
    Integer first_col = rows_index[row];
    for( Integer j=first_col,js=rows_index[row+1]; j<js; ++j ){
      if (columns[j]==row){
        Integer c = columns[first_col];
        Real v = values[first_col];
        columns[first_col] = columns[j];
        values[first_col] = values[j];
        columns[j] = c;
        values[j] = v;
      }
    }
  }
  if (arcaneIsCheck())
    checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixImpl::
assemble()
{
  IntegerConstArrayView rows_index = m_rows_index;
  IntegerArrayView columns = m_columns;
  RealConstArrayView values = m_values;

  SharedArray<Integer> new_rows_index(m_nb_row+1);
  SharedArray<Integer> new_columns;
  SharedArray<Real> new_values;

  new_rows_index[0] = 0;
  for( Integer row=0, nb_row = m_nb_row; row<nb_row; ++row ){
    Integer first_col = rows_index[row];
    for( Integer j=first_col,js=rows_index[row+1]; j<js; ++j ){
      if (columns[j]>=0){
        new_columns.add(columns[j]);
        new_values.add(values[j]);
      }
    }
    new_rows_index[row+1] = new_columns.size();
  }

  m_rows_index = new_rows_index;
  m_columns = new_columns;
  m_values = new_values;
  m_nb_element = new_values.size();
  if (arcaneIsCheck())
    checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatrixImpl::
dump(std::ostream& o)
{
  o << "(Matrix nb_row=" << m_nb_row << " nb_col=" << m_nb_column << ")\n";
  //Integer index = 0;
  for( Integer i=0, is=m_nb_row; i<is; ++i ){
    o << "[" << i << "] ";
    Real sum = 0.0;
    for( Integer z=m_rows_index[i] ,zs=m_rows_index[i+1]; z<zs; ++z ){
      Integer j = m_columns[z];
      Real v = m_values[z];
      sum += v;
      o << " ["<<j<<"]="<<v;
    }
    o << " S=" << sum << '\n';
  }
  //o << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix Matrix::
read(const String& filename)
{
  std::ifstream ifile(filename.localstr());
  Integer nb_x = 0;
  Integer nb_y = 0;
  ifile >> ws >> nb_x >> ws >> nb_y;
  //cout << "** ** N=" << nb_x << ' ' << nb_y << '\n';
  if (nb_x!=nb_y)
    throw FatalErrorException("Matrix::read","not a square matrix");
  Matrix m(nb_x,nb_y);
  IntegerUniqueArray rows_size(nb_x);
  IntegerUniqueArray columns;
  RealUniqueArray values;
  for( Integer x=0; x<nb_x; ++x ){
    Integer nb_column = 0;
    for( Integer y=0; y<nb_y; ++y ){
      Real v =0.0;
      ifile >> v;
      if (!math::isZero(v)){
        columns.add(y);
        values.add(v);
        ++nb_column;
      }
    }
    rows_size[x] = nb_column;
  }
  m.setRowsSize(rows_size);
  m.setValues(columns,values);
  return m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix Matrix::
readHypre(const String& filename)
{
  std::ifstream ifile(filename.localstr());
  Integer nb_x = 0;
  Integer nb_y = 0;
  Integer nb_value = 0;
  Integer xf;
  ifile >> ws >> nb_x >> ws >> nb_y;
  ifile >> ws >> xf >> ws >> xf >> ws >> xf;
  ifile >> ws >> nb_value >> ws >> xf;
  ifile >> ws >> xf >> ws >> xf;
  ifile >> ws >> xf >> ws >> xf;
  //cout << "** ** N=" << nb_x << ' ' << nb_y << '\n';
  if (nb_x!=nb_y)
    throw FatalErrorException(A_FUNCINFO,"not a square matrix");
  Matrix m(nb_x,nb_y);
  IntegerUniqueArray rows_size(nb_x);
  IntegerUniqueArray columns;
  RealUniqueArray values;

  Integer nb_column = 0;
  Integer last_row = 0;
  for( Integer i=0; i<nb_value; ++i ){
    Integer x = 0;
    Integer y = 0;
    Real v = 0.0;
    ifile >> ws >> x >> ws >> y >> ws >> v;
    if (x!=last_row){
      // Nouvelle ligne.
      rows_size[last_row] = nb_column;
      nb_column = 0;
      last_row = x;
    }
    columns.add(y);
    values.add(v);
    ++nb_column;
  }
  rows_size[last_row] = nb_column;
  m.setRowsSize(rows_size);
  m.setValues(columns,values);
  return m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MatVec

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
