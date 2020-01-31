#include "hypre_backend.h"
#include "hypre_matrix.h"

#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Core/Backend/IMatrixConverter.h>
#include <ALIEN/Core/Backend/MatrixConverterRegisterer.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRMatrix.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>

#include <arccore/collections/Array2.h>

class SimpleCSR_to_Hypre_MatrixConverter : public Alien::IMatrixConverter {
public:
  SimpleCSR_to_Hypre_MatrixConverter() {}

  virtual ~SimpleCSR_to_Hypre_MatrixConverter() {}

public:
  BackEndId sourceBackend() const {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name();
  }

  BackEndId targetBackend() const { return Alien::AlgebraTraits<Alien::BackEnd::tag::hypre>::name(); }

  void convert(const Alien::IMatrixImpl *sourceImpl, Alien::IMatrixImpl *targetImpl) const;

  void _build(const Alien::SimpleCSRMatrix<Arccore::Real> &sourceImpl, Alien::Hypre::Matrix &targetImpl) const;

  void _buildBlock(
          const Alien::SimpleCSRMatrix<Arccore::Real> &sourceImpl, Alien::Hypre::Matrix &targetImpl) const;
};

void
SimpleCSR_to_Hypre_MatrixConverter::convert(
        const IMatrixImpl *sourceImpl, IMatrixImpl *targetImpl) const {
  const auto &v = cast<Alien::SimpleCSRMatrix<Arccore::Real> >(sourceImpl, sourceBackend());
  auto &v2 = cast<Alien::Hypre::Matrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting Alien::SimpleCSRMatrix: " << &v << " to Hypre::Matrix " << &v2;
  });
  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void
SimpleCSR_to_Hypre_MatrixConverter::_build(
        const Alien::SimpleCSRMatrix<Arccore::Real> &sourceImpl, Alien::Hypre::Matrix &targetImpl) const {
  const auto &dist = sourceImpl.distribution();
  const auto &profile = sourceImpl.getCSRProfile();
  const auto localSize = profile.getNRow();
  const auto localOffset = dist.rowOffset();
  const auto &matrixInternal = sourceImpl.internal();

  auto data_count = 0;
  auto pos = 0;
  auto max_line_size = 0;
  Arccore::UniqueArray<int> sizes(localSize);
  for (auto row = 0; row < localSize; ++row) {
    data_count += profile.getRowSize(row);
    sizes[pos] = profile.getRowSize(row);
    max_line_size = std::max(max_line_size, profile.getRowSize(row));
    ++pos;
  }

  int ilower = localOffset;
  int iupper = localOffset + localSize - 1;
  int jlower = ilower;
  int jupper = iupper;

  alien_debug([&] {
    cout() << "Matrix range : "
           << "[" << ilower << ":" << iupper << "]"
           << "x"
           << "[" << jlower << ":" << jupper << "]";
  });
  {

    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
    }

    // Buffer de construction
    Arccore::UniqueArray<double> values(std::max(localSize, max_line_size));
    auto &indices = sizes; // réutilisation du buffer
    indices.resize(std::max(localSize, max_line_size));

    auto m_values = matrixInternal.getValues();
    auto cols = profile.getCols();
    auto icount = 0;
    for (auto irow = 0; irow < localSize; ++irow) {
      int row = localOffset + irow;
      int ncols = profile.getRowSize(irow);
      auto jpos = 0;
      for (auto k = 0; k < ncols; ++k) {
        indices[jpos] = cols[icount];
        values[jpos] = m_values[icount];
        ++jpos;
        ++icount;
      }

      const bool success =
              targetImpl.setMatrixValues(1, &row, &ncols, indices.data(), values.data());

      if (not success) {
        throw Arccore::FatalErrorException(
                A_FUNCINFO, Arccore::String::format("Cannot set Hypre Matrix Values for row {0}", row));
      }
    }
  }
  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}

void
SimpleCSR_to_Hypre_MatrixConverter::_buildBlock(
        const Alien::SimpleCSRMatrix <Arccore::Real> &sourceImpl, Alien::Hypre::Matrix &targetImpl) const {
  const auto &dist = sourceImpl.distribution();
  const auto &profile = sourceImpl.getCSRProfile();
  const auto localSize = profile.getNRow();
  const auto block_size = targetImpl.block()->size();
  const auto localOffset = dist.rowOffset();
  const auto &matrixInternal = sourceImpl.internal();

  auto max_line_size = localSize * block_size;
  auto data_count = 0;
  auto pos = 0;
  Arccore::UniqueArray<int> sizes(localSize * block_size);
  for (auto row = 0; row < localSize; ++row) {
    auto row_size = profile.getRowSize(row) * block_size;
    for (auto ieq = 0; ieq < block_size; ++ieq) {
      data_count += row_size;
      sizes[pos] = row_size;
      ++pos;
    }
    max_line_size = std::max(max_line_size, row_size);
  }

  int ilower = localOffset * block_size;
  int iupper = (localOffset + localSize) * block_size - 1;
  int jlower = ilower;
  int jupper = iupper;

  alien_debug([&] {
    cout() << "Matrix range : "
           << "[" << ilower << ":" << iupper << "]"
           << "x"
           << "[" << jlower << ":" << jupper << "]";
  });

  // Buffer de construction
  Arccore::UniqueArray2 <Arccore::Real> values;
  values.resize(block_size, max_line_size);
  Arccore::UniqueArray<int> &indices = sizes; // réutilisation du buffer
  indices.resize(std::max(max_line_size, localSize * block_size));
  // est ce qu'on reconstruit la matrice  ?
  {
    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
    }

    auto cols = profile.getCols();
    auto m_values = matrixInternal.getValues();
    auto col_count = 0;
    auto mat_count = 0;
    for (auto irow = 0; irow < localSize; ++irow) {
      int row = localOffset + irow;
      int ncols = profile.getRowSize(irow);
      auto jcol = 0;
      for (auto k = 0; k < ncols; ++k)
        for (auto j = 0; j < block_size; ++j)
          indices[jcol++] = cols[col_count + k] * block_size + j;
      for (auto k = 0; k < ncols; ++k) {
        const auto kk = k * block_size * block_size;
        for (auto i = 0; i < block_size; ++i)
          for (auto j = 0; j < block_size; ++j)
            values[i][k * block_size + j] = m_values[mat_count + kk + i * block_size + j];
      }
      col_count += ncols;
      mat_count += ncols * block_size * block_size;

      for (auto i = 0; i < block_size; ++i) {
        auto rows = row * block_size + i;
        auto num_cols = ncols * block_size;
        const bool success = targetImpl.setMatrixValues(
                1, &rows, &num_cols, indices.data(), values[i].data());

        if (not success) {
          throw Arccore::FatalErrorException(
                  A_FUNCINFO, Arccore::String::format("Cannot set Hypre Matrix Values for row {0}", row));
        }
      }
    }
  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_Hypre_MatrixConverter);
