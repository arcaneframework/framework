#include <gtest/gtest.h>

#include <alien/kernels/dok/DoKMatrixT.h>
#include <alien/distribution/MatrixDistribution.h>

// For DoKReverseIndexer test.
#include <alien/kernels/dok/DoKReverseIndexer.h>

// For SimpleCSR convert tests
#include <alien/kernels/dok/converters/to_simple_csr_matrix.h>
#include <alien/kernels/dok/converters/from_simple_csr_matrix.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/dok/DoKBackEnd.h>

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

namespace {
Arccore::Int32 g_row[] = { 0, 0, 1, 1, 1, 2, 2 };
Arccore::Int32 g_col[] = { 0, 1, 0, 1, 2, 0, 2 };
Arccore::Real g_value[] = { 3, -2, -1, 4, -0.5, -2, 5 };
}

/*
 * Auxiliary class to build toy matrices.
 */
class TestDoKBuilder
{
 public:
  TestDoKBuilder(int num_rows = 3, bool square = false)
  : m_num_rows(num_rows)
  , m_num_cols(num_rows)
  , m_mdist(m_num_rows, m_num_cols, Environment::parallelMng())
  , m_row_space(new Alien::Space(m_mdist.globalRowSize(), "RowSpace"))
  , m_col_space(new Alien::Space(m_mdist.globalColSize(), "ColSpace"))
  , m_multimat()
  {
    if (square)
      m_col_space = m_row_space;
    m_multimat.reset(
        new Alien::MultiMatrixImpl(m_row_space, m_col_space, m_mdist.clone()));
  }

  template <class Matrix> Matrix* createEmptyMatrix() const
  {
    return new Matrix(m_multimat.get());
  }

  void fill(Alien::DoKMatrix& mat) const
  {
    int lenght = sizeof(g_row) / sizeof(Arccore::Int32);
    for (int i = 0; i < lenght; i++) {
      if (g_row[i] >= row_begin() && g_row[i] < row_end())
        //mat.setMatrixValue(g_row[i], g_col[i], g_value[i]);
        mat.setNNZ(g_row[i], g_col[i], g_value[i]);
    }
  }

  Arccore::Int32 row_begin() const { return m_mdist.rowOffset(); }

  Arccore::Int32 row_end() const { return m_mdist.rowOffset() + m_mdist.localRowSize(); }

 private:
  Arccore::Int32 m_num_rows;
  Arccore::Int32 m_num_cols;
  Alien::MatrixDistribution m_mdist;
  std::shared_ptr<Alien::Space> m_row_space;
  std::shared_ptr<Alien::Space> m_col_space;
  std::unique_ptr<Alien::MultiMatrixImpl> m_multimat;
};

TEST(TestDoKMatrix, DoKReverseIndexer)
{
  Alien::DoKReverseIndexer r_index;

  std::cout << std::is_same<int, Arccore::Int32>::value << std::endl;
  std::cout << std::is_same<long long, Arccore::Int64>::value << std::endl;
  // std::cout << std::is_same<long long, Alien::DoKReverseIndexer::Index>::value <<
  // std::endl;

  std::pair<int, int> i1(0, 0);
  long long o(0);
  r_index.record(o, i1);

  Alien::DoKReverseIndexer::Index i2(1, 0);
  r_index.record(1, i2);
  Alien::DoKReverseIndexer::Index i3(0, 1);
  r_index.record(2, i3);
  Alien::DoKReverseIndexer::Index i4(1, 1);
  r_index.record(3, i4);

  ASSERT_EQ(i1, r_index[0]);
  ASSERT_EQ(i2, r_index[1]);
  ASSERT_EQ(i3, r_index[2]);
  ASSERT_EQ(i4, r_index[3]);
}

TEST(TestDoKMatrix, Constructor)
{
  TestDoKBuilder builder;
  std::unique_ptr<Alien::DoKMatrix> mat(builder.createEmptyMatrix<Alien::DoKMatrix>());

  builder.fill(*mat);
  mat->compact();
  mat->assemble();

  auto& data = mat->data();
  auto r_index = data.getReverseIndexer();

  auto start = builder.row_begin();
  auto stop = builder.row_end();
  for (int i = 0; i < r_index->size(); i++) {
    auto cur = (*r_index)[i];
    //ASSERT_GE(cur.first, start);
    //ASSERT_LT(cur.first, stop);
    ASSERT_GE(cur.value().first, start);
    ASSERT_LT(cur.value().first, stop);
  }
}

TEST(TestDoKMatrix, ConvertFromCSR)
{
  Arccore::Int32 rows = 5;
  Arccore::Int32 cols = 5;
  Alien::MatrixDistribution mdist(rows, cols, Environment::parallelMng());
  Alien::Space row_space(mdist.globalRowSize(), "Space");
  Alien::Space col_space(mdist.globalColSize(), "Space");

  std::unique_ptr<Alien::MultiMatrixImpl> multimat(
      new Alien::MultiMatrixImpl(row_space.clone(), col_space.clone(), mdist.clone()));

  auto& csr_mat(multimat->get<Alien::BackEnd::tag::simplecsr>());
  auto& dok_mat(multimat->get<Alien::BackEnd::tag::DoK>(true));

  Alien::SimpleCSRtoDoKMatrixConverter converter;
  converter.convert(&csr_mat, &dok_mat);

  dok_mat.assemble();
}

TEST(TestDoKMatrix, ConvertToCSR)
{
  TestDoKBuilder builder(3, true);
  std::unique_ptr<Alien::DoKMatrix> dok_mat(
      builder.createEmptyMatrix<Alien::DoKMatrix>());

  builder.fill(*dok_mat);
  dok_mat->compact();

  //dok_mat->setMatrixValue(0, 2, -5.);
  dok_mat->setNNZ(0, 2, -5.);

  dok_mat->assemble();

  typedef Alien::SimpleCSRMatrix<Arccore::Real> SimpleCSR;
  std::unique_ptr<SimpleCSR> csr_mat(builder.createEmptyMatrix<SimpleCSR>());
  Alien::DoKtoSimpleCSRMatrixConverter converter;
  converter.convert(dok_mat.get(), csr_mat.get());
}

// CEA MatrixData and DirectMatrixBuilder are required for the next test.
#if 0
TEST(TestDoKMatrix, MultiImplConverter)
{
  // Build a SimpleCSR matrix
  Alien::MatrixDistribution mdist(4, 4, Environment::parallelMng());
  Alien::Space row_space(4, "Space");
  Alien::Space col_space(4, "Space");
  Alien::MatrixData A(row_space, col_space, mdist);
  ASSERT_EQ(A.rowSpace(), row_space);
  ASSERT_EQ(A.colSpace(), col_space);
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  Alien::DirectMatrixBuilder builder(std::move(A), tag);
  builder.reserve(5);
  builder.allocate();

  Alien::Integer first = mdist.rowOffset();
  Alien::Integer last = first + mdist.localRowSize();

  if ( first <= 0 && 0 < last)
    builder(0, 0) = -1.;
  if (first <= 1 && 1 < last)
    builder(1, 1) = -2.;
  if (first <= 2 && 2 < last) {
    builder(2, 2) = -3.;
    builder(2, 3) = 3.14;
  }
  if (first <= 3 && 3 < last) {
    builder(3, 1) = 2.71;
    builder(3, 3) = -4;
  }
  builder.finalize();

  std::cerr << builder.stats() << std::endl;

  A = builder.release();



  Alien::MultiMatrixImpl* multiA = A.impl();
  const Alien::DoKMatrix& dok_a = multiA->get<Alien::BackEnd::tag::DoK>();
  dok_a.backend();
}
#endif // 0
