// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatVecUnitTest.cc                                           (C) 2000-2026 */
/*                                                                           */
/* Service de test des matrices/vecteurs.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/core/Directory.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"

#include "arcane/matvec/Matrix.h"
#include "arcane/matvec/Vector.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MatVecUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace MatVec;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du maillage
 */
class MatVecUnitTest
: public ArcaneMatVecUnitTestObject
{
 public:

  explicit MatVecUnitTest(const ServiceBuildInfo& cb);
  ~MatVecUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  VariableCellReal m_pressure;
  VariableCellInteger m_cell_matrix_row;
  VariableCellInteger m_cell_matrix_column;

  void _testArcaneMatrix1();
  void _testArcaneMatrix2();
  void _testArcaneMatrix3();
  void _initPressure();
  void _initMatrix();
  void _testMatrix2();
  void _printResidualInfo(const Matrix& matrix, const Vector& vector_b,
                          const Vector& vector_x);
  void _buildPoissonMatrixAndRHS(Matrix& matrix, Vector& rhs_vector, Int32 cube_size);
  void _testPoissonMatrix(Int32 cube_size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MATVECUNITTEST(MatVecUnitTest,MatVecUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatVecUnitTest::
MatVecUnitTest(const ServiceBuildInfo& mb)
: ArcaneMatVecUnitTestObject(mb)
, m_pressure(VariableBuildInfo(mb.mesh(),"MatVecPressure"))
, m_cell_matrix_row(VariableBuildInfo(mb.mesh(),"MatVecCellMatrixRow"))
, m_cell_matrix_column(VariableBuildInfo(mb.mesh(),"MatVecCellMatrixColumn"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatVecUnitTest::
~MatVecUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
executeTest()
{
  info() << "** EXEC TEST";
  //_testArcaneMatrix2();
  //_initMatrix();
  _testArcaneMatrix3();
  //_testMatrix2();
  _testPoissonMatrix(4);
  _testPoissonMatrix(8);
  //_testPoissonMatrix(16);
  //_testPoissonMatrix(24);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
initializeTest()
{
  info() << "** INIT TEST";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
_testArcaneMatrix1()
{
  using namespace MatVec;
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
  Real epsilon = 1.0e-8;
  {
#if 0
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
#endif
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

void MatVecUnitTest::
_initPressure()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
_printResidualInfo(const Matrix& a,const Vector& b,const Vector& x)
{
  OStringStream ostr;
  Vector tmp(b.size());
  // tmp = b - Ax
  MatrixOperation mat_op;
  mat_op.matrixVectorProduct(a,x,tmp);
  //ostr() << "\nAX=";
  //tmp.dump(ostr());
  mat_op.negateVector(tmp);
  mat_op.addVector(tmp,b);
  Real r = mat_op.dot(tmp);
  info() << " RESIDUAL NORM="  << r;

  //ostr() << "\nR=";
  //tmp.dump(ostr());
  //info() << " RESIDUAL NORM="  << r <<  " RESIDUAL=" << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
_initMatrix()
{
  IItemFamily* cell_family = mesh()->cellFamily();
  CellGroup cells = cell_family->allItems().own();
  Integer nb_cell = cells.size();

  Integer nb_row = nb_cell;
  Integer nb_column = nb_cell;

  m_cell_matrix_row.fill(-1);
  m_cell_matrix_column.fill(-1);
  {
    Integer index = 0;;
    ENUMERATE_CELL(icell,cells){
      //const Cell& cell = *icell;
      m_cell_matrix_row[icell] = index;
      m_cell_matrix_column[icell] = index; // Uniquement valable en sequentiel
      ++index;
    }
  }


  IntegerUniqueArray rows_size(nb_cell);
  rows_size.fill(0);
  IntegerUniqueArray columns;
  RealUniqueArray values;
  {
    ENUMERATE_CELL(icell,cells){
      const Cell& cell = *icell;
      //Integer nb_face = cell.nbFace();
      Integer cell_row = m_cell_matrix_row[icell];
      Integer nb_col = 0;
      for( Face face : cell.faces() ){
        Integer face_nb_cell = face.nbCell();
        if (face_nb_cell==2){
          Cell opposite_cell = (face.cell(0)==cell) ? face.cell(1) : face.cell(0);
          columns.add(m_cell_matrix_column[opposite_cell]);
          values.add(-1.0);
          ++nb_col;
        }
      }
      columns.add(m_cell_matrix_column[icell]);
      values.add(10.0+(Real)(nb_col));
      ++nb_col;
      rows_size[cell_row] = nb_col;
    }
  }

  MatVec::Matrix matrix(nb_row,nb_column);
  matrix.setRowsSize(rows_size);
  matrix.setValues(columns,values);
  Integer nb_element = values.size();
  info() << "** MATRIX NB_ELEMENT=" << nb_element;
  if (nb_element<500){
    OStringStream ostr;
    matrix.dump(ostr());
    info() << " MATRIX = " << ostr.str();
  }
  MatVec::Vector rhs(nb_row);
  for(Integer i=0; i<nb_row; ++i ){
    rhs.values()[i] = (Real)(i*10+1);
    //rhs.values()[i] = 2.0;
  }
  //{
  //OStringStream ostr;
  //rhs.dump(ostr());
  //info() << " VECTOR_B= " << ostr.str();
  //}
  Real epsilon = 1.0e-7;

  if (1){
    //AMGSolver amg(traceMng());
    //amg.build(matrix);

    MatVec::Vector x(nb_row);
    x.values().fill(0.0);

    info() << "SOLVE USING AMG PRECONDITIONNER ";

    AMGPreconditioner amg_preconditioner(traceMng());
    amg_preconditioner.build(matrix);
    MatVec::ConjugateGradientSolver cgs;
    cgs.solve(matrix,rhs,x,epsilon,&amg_preconditioner);
    OStringStream ostr;
    x.dump(ostr());
    info() << "SOLVE FULL_PRECONDITIONNER NB_ITERATION=" << cgs.nbIteration()
           << " norm=" << cgs.residualNorm();
    //<< " X=" << ostr.str();
    _printResidualInfo(matrix,rhs,x);
  }

  if (1){
    AMGSolver amg(traceMng());
    amg.build(matrix);

    MatVec::Vector x(nb_row);
    x.values().fill(0.0);

    info() << "SOLVE USING AMG ";

    for( Integer nb_iter=0; nb_iter<5; ++nb_iter ){
      //OStringStream ostr;
      amg.solve(rhs,x);
      //x.dump(ostr());
      //info() << "SOLVE USING AMG " << nb_iter << " X=" << ostr.str();
      info() << "SOLVE USING AMG " << nb_iter;
      _printResidualInfo(matrix,rhs,x);
    }
  }

  if (0){
    AMGSolver amg(traceMng());
    amg.build(matrix);

    MatVec::Vector x(nb_row);
    x.values().fill(0.0);

    info() << "SOLVE USING AMG ONE_ITERATION";
    amg.solve(rhs,x);
    _printResidualInfo(matrix,rhs,x);

    MatVec::ConjugateGradientSolver cgs;
    cgs.solve(matrix,rhs,x,epsilon,0);
    OStringStream ostr;
    x.dump(ostr());
    info() << "SOLVE AMG ONE_ITERATION NB_ITERATION=" << cgs.nbIteration()
           << " norm=" << cgs.residualNorm()
           << " X=" << ostr.str();
  }

  if (0){
    MatVec::Vector x(nb_row);
    x.values().fill(0.0);
    MatVec::ConjugateGradientSolver cgs;
    cgs.solve(matrix,rhs,x,epsilon,0);
    OStringStream ostr;
    x.dump(ostr());
    info() << "SOLVE CONJUGATE_GRADIENT NB_ITERATION=" << cgs.nbIteration()
           << " norm=" << cgs.residualNorm()
           << " X=" << ostr.str();
    _printResidualInfo(matrix,rhs,x);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
_testMatrix2()
{
  info() << "TEST DIAGONAL CONJUGATE GRADIENT";
  Matrix m(6,6);
  IntegerUniqueArray rows(6);
  //for( Integer i=0; i<6; ++i )
  //rows[i] = 6;
  rows[0] = 2;
  rows[1] = 3;
  rows[2] = 3;
  rows[3] = 3;
  rows[4] = 3;
  rows[5] = 2;
  m.setRowsSize(rows);
   
  // Normalement cette matrice converge tres difficilement
  // voire ne converge pas.
  m.setValue(0,0, 1.31814608610996E-12);
  m.setValue(0,1,-1.31814608610859E-12);

  m.setValue(1,0,-1.31814608610859E-12);
  m.setValue(1,1, 4.66093078486141E-11);
  m.setValue(1,2,-4.5291161762499E-11);
    
  m.setValue(2,1,-4.5291161762499E-11);
  m.setValue(2,2,1.95243397695228E-10);
  m.setValue(2,3,-1.49952235932723E-10);

  m.setValue(3,2,-1.49952235932723E-10);
  m.setValue(3,3,4.31954546380107E-10);
  m.setValue(3,4,-2.82002310447369E-10);
    
  m.setValue(4,3,-2.82002310447369E-10);
  m.setValue(4,4,3.27500080633377E-08);
  m.setValue(4,5,-3.24680057528885E-08);
    
  m.setValue(5,4,-3.24680057528885E-08);
  m.setValue(5,5,3.24680057528886E-08);

  //m.Dump(Console.Out);
  //Console.WriteLine();
  //_PrintMatrix(m);

  Vector b(6);
  b.values()[0] =  2.883E+014;
  b.values()[1] = -1.745E+016;
  b.values()[2] =  5.996E+015;
  b.values()[3] = -1.530E+017;
  b.values()[4] = -1.021E+019;
  b.values()[5] =  9.716E+018;

  //Vector xref = new Vector(6);
  //xref.Values[0] = 
  Vector x(6);
  //for( int i=0; i<6; ++i )
  //x.Values[i] = 0.0;
  {
    x.values().fill(0.0);
    ConjugateGradientSolver cg;
    DiagonalPreconditioner diag(m);
    cg.solve(m,b,x,1e-12,&diag);
    info() << String::format("DIAG NB ITER={0} RESIDU={1}",cg.nbIteration(),cg.residualNorm());
  }
  {
    x.values().fill(0.0);
    ConjugateGradientSolver cg;
    AMGPreconditioner amg_preconditioner(traceMng());
    amg_preconditioner.build(m);
    cg.solve(m,b,x,1e-12,&amg_preconditioner);
    info() << String::format("AMG NB ITER={0} RESIDU={1}",cg.nbIteration(),cg.residualNorm());
  }


  //Console.WriteLine("X=");
  //x.Dump(Console.Out);
  //Console.WriteLine();

  /*Console.WriteLine("B=");
    b.Dump(Console.Out);
    Console.WriteLine();

    Vector b2 = new Vector(6);
    MatrixOperation mo = new MatrixOperation();
    mo.MatrixVectorProduct(m,x,b2);
    Console.WriteLine("AX=");
    b2.Dump(Console.Out);
    Console.WriteLine();
    Vector s = new Vector(6);
    s.Copy(b2);
    mo.NegateVector(s);
    mo.AddVector(s,b2);
    Console.WriteLine("B-AX=");
    s.Dump(Console.Out);
    Console.WriteLine();*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
_testArcaneMatrix3()
{
  Directory base_dir("hypre_small_poisson");

  String matrix_path = base_dir.file("IJ.out.A.00000");

  info() << "Reading matrix path=" << matrix_path;
  Matrix matrix(Matrix::readHypre(matrix_path));
  const Int32 nb_row = matrix.nbRow();

  String vector_b_path = base_dir.file("IJ.out.b.00000");
  info() << "Reading vector path=" << vector_b_path;
  Vector vector_b(Vector::readHypre(vector_b_path));

  String vector_x_path = base_dir.file("IJ.out.x.00000");
  info() << "Reading vector path=" << vector_b_path;
  Vector vector_x_ref(Vector::readHypre(vector_x_path));

  Vector vector_x(nb_row);
  vector_x.values().fill(0.0);

  {
    ConjugateGradientSolver cg;
    //DiagonalPreconditioner diag(matrix);

    AMGPreconditioner amg_preconditioner(traceMng());
    amg_preconditioner.build(matrix);

    cg.solve(matrix,vector_b,vector_x,1e-12,&amg_preconditioner);
    auto vx = vector_x.values();
    for (Int32 i = 0; i < nb_row; ++i) {
      Real v = vx[i];
      Real v_ref = vector_x_ref.values()[i];
      Real diff = (v - v_ref);
      if (v_ref != 0.0)
        diff /= v_ref;
      info() << "X[" << i << "] = " << v << " ref=" << v_ref << " diff=" << diff;
    }
    info() << String::format("DIAG NB ITER={0} RESIDU={1}", cg.nbIteration(), cg.residualNorm());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
_testArcaneMatrix2()
{
  using namespace MatVec;


  Matrix m1(Matrix::read("test.mat"));
  MatrixOperation2 mat_op2;
  Matrix m2(m1.clone());
  Matrix m3 = mat_op2.matrixMatrixProduct(m1,m2);  
  Matrix m4 = mat_op2.matrixMatrixProductFast(m1,m2);
  OStringStream ostr;
  ostr() << "\nLEFT_MATRIX=";
  m1.dump(ostr());
  ostr() << "\nRIGHT_MATRIX=";
  m2.dump(ostr());
  ostr() << "\nPRODUCT_MATRIX=";
  m3.dump(ostr());
  ostr() << "\nPRODUCT_MATRIX_FAST=";
  m4.dump(ostr());
  info() << " DUMP PRODUCT MATRIX=" << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Génère une matrice et le vector solution pour une
 * équation de poisson 3D sur un cube unité.
 *
 * \a cube_size est le nombre d'inconnues par côté. La taille de la matrice
 * est donc `cube_size ^ 3`.
 */
void MatVecUnitTest::
_buildPoissonMatrixAndRHS(Matrix& matrix, Vector& rhs_vector, Int32 cube_size)
{
  UniqueArray<Real> val;
  UniqueArray<Int32> col;
  UniqueArray<Real> rhs;

  double anisotropy = 1.0;

  const Int32 n2 = cube_size * cube_size;
  const Int32 n3 = n2 * cube_size;

  col.reserve(n3 * 7);
  val.reserve(n3 * 7);
  rhs.reserve(n3);

  const Real hx = 1.0;
  const Real hy = hx * anisotropy;
  const Real hz = hy * anisotropy;
  const Real hx2 = hx * hx;
  const Real hy2 = hy * hy;
  const Real hz2 = hz * hz;

  UniqueArray<Int32> rows_size;
  rows_size.reserve(n3);
  Int32 last_column_index = 0;
  Int32 index = 0;
  for (Int32 k = 0; k < cube_size; ++k) {
    for (Int32 j = 0; j < cube_size; ++j) {
      for (Int32 i = 0; i < cube_size; ++i, ++index) {
        if (k > 0) {
          col.add(index - n2);
          val.add(-1.0 / hz2);
        }

        if (j > 0) {
          col.add(index - cube_size);
          val.add(-1.0 / hy2);
        }

        if (i > 0) {
          col.add(index - 1);
          val.add(-1.0 / hx2);
        }

        col.add(index);
        val.add((2 / hx2 + 2 / hy2 + 2 / hz2));

        if (i + 1 < cube_size) {
          col.add(index + 1);
          val.add(-1.0 / hx2);
        }

        if (j + 1 < cube_size) {
          col.add(index + cube_size);
          val.add(-1.0 / hy2);
        }

        if (k + 1 < cube_size) {
          col.add(index + n2);
          val.add(-1.0 / hz2);
        }

        rhs.add(1.0);
        Int32 nb_row = col.size() - last_column_index;
        //info() << "NB_ROW=" << nb_row;
        rows_size.add(nb_row);
        last_column_index = col.size();
      }
    }
  }
  Int32 nb_row = rows_size.size();
  Matrix mat(nb_row, nb_row);
  mat.setRowsSize(rows_size);
  mat.setValues(col, val);
  matrix = mat;

  Vector vec(nb_row);
  vec.values().copy(rhs);
  rhs_vector = vec;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatVecUnitTest::
_testPoissonMatrix(const Int32 cube_size)
{
  info() << "Testing 3D Poisson Matrix cube_size=" << cube_size;

  Matrix matrix;
  Vector rhs;
  _buildPoissonMatrixAndRHS(matrix, rhs, cube_size);
  Int32 nb_row = matrix.nbRow();

  ConjugateGradientSolver cg;

  Vector x_diag(nb_row);
  ArrayView<Real> x_diag_view(x_diag.values());
  x_diag_view.fill(0.0);

  {
    DiagonalPreconditioner diag(matrix);
    cg.solve(matrix, rhs, x_diag, 1e-12, &diag);
    info() << String::format("DIAG NB_ROW={0} NB ITER={1} RESIDU={2}", nb_row, cg.nbIteration(), cg.residualNorm());
  }

  Vector x_amg(nb_row);
  ArrayView<Real> x_amg_view(x_amg.values());
  x_amg_view.fill(0.0);

  {
    AMGPreconditioner amg_preconditioner(traceMng());
    amg_preconditioner.build(matrix);
    cg.solve(matrix, rhs, x_amg, 1e-12, &amg_preconditioner);
    info() << String::format("AMG NB_ROW={0} NB ITER={1} RESIDU={2}", nb_row, cg.nbIteration(), cg.residualNorm());
  }

  Vector x2(nb_row);
  ArrayView<Real> x2_view(x2.values());
  x2_view.fill(0.0);
  if (nb_row < 500) {
    DirectSolver direct_solver;
    direct_solver.solve(matrix, rhs, x2);
  }
  for (Int32 i = 0; i < 10; ++i) {
    Real v_diag = x_diag_view[i];
    Real v_amg = x_amg_view[i];
    Real v_ref = x2_view[i];
    info() << "I=" << i << " X_DIAG=" << v_diag << " X_AMG=" << v_amg << " REF=" << v_ref
           << " diff_diag=" << (v_diag - v_ref) << " diff_amg=" << (v_amg - v_ref)
           << " diff_amg_diag=" << (v_diag - v_amg);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
