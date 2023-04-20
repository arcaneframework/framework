
#include <list>
#include <vector>
#include <sstream>

#include <alien/kernels/hpddm/HPDDMPrecomp.h>

#include <alien/expression/solver/SolverStater.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <alien/kernels/hpddm/data_structure/HPDDMInternal.h>

#include <alien/kernels/hpddm/data_structure/HPDDMMatrix.h>

#include <alien/kernels/hpddm/HPDDMBackEnd.h>

#include <alien/core/impl/MultiMatrixImpl.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
/*---------------------------------------------------------------------------*/
BEGIN_HPDDMINTERNAL_NAMESPACE

bool HPDDMInternal::m_is_initialized = false;

template <>
int
HPDDMInternal::getEnv(std::string const& key, int default_value)
{
  const char* arch_str = ::getenv(key.c_str());
  if (arch_str)
    return atoi(arch_str);
  return default_value;
}

void
HPDDMInternal::initialize(IMessagePassingMng* parallel_mng)
{
  if (m_is_initialized)
    return;
  m_is_initialized = true;
}

void
HPDDMInternal::finalize()
{
  m_is_initialized = false;
}

template <typename ValueT>
void
MatrixInternal<ValueT>::_computeMPIGhostMatrix(
    typename MatrixInternal<ValueT>::CSRMatrixType const& A)
{
  using namespace Arccore;
  auto const& profile = A.getProfile();
  auto const& dist_info = A.getDistStructInfo();

  auto kcol = profile.kcol();
  auto cols_A = profile.cols(); // cols_uid
  auto& cols = dist_info.m_cols; // cols_lid
  auto values = A.getAddressData();

  auto const& send_info = dist_info.m_send_info;
  std::size_t num_neighbours_s = dist_info.m_send_info.m_num_neighbours;
  auto const& ids_offset_s = dist_info.m_send_info.m_ids_offset;
  auto const& ids_s = dist_info.m_send_info.m_ids;

  auto const& recv_info = dist_info.m_recv_info;
  std::size_t num_neighbours_r = dist_info.m_recv_info.m_num_neighbours;
  auto const& ids_offset_r = dist_info.m_recv_info.m_ids_offset;
  auto const& uids_r = dist_info.m_recv_info.m_uids;

  UniqueArray<MessagePassing::Request> send_request0(num_neighbours_s);
  UniqueArray<MessagePassing::Request> send_request1(num_neighbours_s);

  UniqueArray<int> nnz_ext_int_send(num_neighbours_s);
  UniqueArray<int> nnz_ext_ext_send(num_neighbours_s);
  UniqueArray<int> nnz_ext_ext2_send(num_neighbours_s);

  UniqueArray<int> nrows_send(num_neighbours_s);
  UniqueArray<int> count_send(4 * num_neighbours_s);

  {
    for (std::size_t i = 0; i < num_neighbours_s; ++i) {
      int ranki = dist_info.m_send_info.m_ranks[i];

      int i_recv_neighb = dist_info.m_recv_info.getRankNeighbId(ranki);
      assert(i_recv_neighb != -1);

      nnz_ext_int_send[i] = 0;
      nnz_ext_ext_send[i] = 0;
      nnz_ext_ext2_send[i] = 0;

      for (int j = ids_offset_s[i]; j < ids_offset_s[i + 1]; ++j) {
        int lid = ids_s[j];

        for (int k = kcol[lid]; k < kcol[lid + 1]; ++k) {
          int col = cols[k];
          if (col < m_local_nrows) {
            if (dist_info.isInterfaceRow(col)) {
              // CHECK IF IN IN OVERLAP OF DOMAIN I
              if (send_info.getLocalId(i, col) != -1)
                ++nnz_ext_ext_send[i];
            }
          } else {
            if (col >= ids_offset_r[i_recv_neighb]
                && col < ids_offset_r[i_recv_neighb + 1])
              ++nnz_ext_int_send[i];
            else
              ++nnz_ext_ext2_send[i];
          }
        }
      }
      nrows_send[i] = ids_offset_s[i + 1] - ids_offset_s[i];
      count_send[4 * i] = nrows_send[i];
      count_send[4 * i + 1] = nnz_ext_int_send[i];
      count_send[4 * i + 2] = nnz_ext_ext_send[i];
      count_send[4 * i + 3] = nnz_ext_ext2_send[i];

      send_request0[i] = Arccore::MessagePassing::mpSend(
          m_parallel_mng, ConstArrayView<int>(4, &count_send[4 * i]), ranki, false);
    }
  }

  UniqueArray<int> nnz_ext_int(num_neighbours_r);
  UniqueArray<int> nnz_ext_ext(num_neighbours_r);
  UniqueArray<int> nnz_ext_ext2(num_neighbours_r);
  UniqueArray<int> nrows(num_neighbours_r);
  UniqueArray<int> count_recv(4 * num_neighbours_r);

  UniqueArray<MessagePassing::Request> recv_request0(num_neighbours_r);
  for (std::size_t i = 0; i < num_neighbours_r; ++i) {
    int ranki = dist_info.m_recv_info.m_ranks[i];
    recv_request0[i] = Arccore::MessagePassing::mpReceive(
        m_parallel_mng, ArrayView<int>(4, &count_recv[4 * i]), ranki, false);
  }

  Arccore::MessagePassing::mpWaitAll(m_parallel_mng, send_request0);
  Arccore::MessagePassing::mpWaitAll(m_parallel_mng, recv_request0);

  UniqueArray<UniqueArray<Integer>> ibuffer_s(num_neighbours_s);
  UniqueArray<UniqueArray<ValueT>> rbuffer_s(num_neighbours_s);
  {
    for (std::size_t i = 0; i < num_neighbours_s; ++i) {

      int ranki = dist_info.m_send_info.m_ranks[i];

      // INIT WORK ARRAY
      int i_recv_neighb = dist_info.m_recv_info.getRankNeighbId(ranki);
      assert(i_recv_neighb != -1);

      std::size_t send_size = nrows_send[i];

      ibuffer_s[i].resize(3 * (send_size + 1) + nnz_ext_int_send[i] + nnz_ext_ext_send[i]
          + 2 * nnz_ext_ext2_send[i]);
      rbuffer_s[i].resize(
          nnz_ext_int_send[i] + nnz_ext_ext_send[i] + nnz_ext_ext2_send[i] + send_size);

      int ext_int_ptr = 0;
      int ext_ext_ptr = 0;
      int ext_ext2_ptr = 0;

      int ext_int_offset = 0;
      int ext_ext_offset = ext_int_offset + nnz_ext_int_send[i];
      int ext_ext2_offset = ext_ext_offset + nnz_ext_ext_send[i];
      int mpi_diag_correction_offset = ext_ext2_offset + nnz_ext_ext2_send[i];

      for (int j = 0; j < ids_offset_s[i + 1] - ids_offset_s[i]; ++j) {
        int lid = ids_s[j + ids_offset_s[i]];
        // mpi_ext_ext_kcol_s[i][j] = ext_ext_ptr;

        ibuffer_s[i][j] = ext_int_ptr;
        ibuffer_s[i][(send_size + 1) + j] = ext_ext_ptr;
        ibuffer_s[i][2 * (send_size + 1) + j] = ext_ext2_ptr;

        // mpi_diag_correction_s[i][j] = 0.;
        rbuffer_s[i][mpi_diag_correction_offset + j] = 0;
        for (int k = kcol[lid]; k < kcol[lid + 1]; ++k) {
          int col = cols[k];
          if (col < m_local_nrows) {
            if (!dist_info.isInterfaceRow(col)) {
              // mpi_diag_correction_s[i][j] += values[k];
              rbuffer_s[i][mpi_diag_correction_offset + j] += values[k];
            } else // if(dist_info.isInterfaceRow(col))
            {
              // CHECK IF IN IN OVERLAP OF DOMAIN I
              int col_lid = send_info.getLocalId(i, col);
              if (col_lid == -1) {
                // mpi_diag_correction_s[i][j] += values[k];
                rbuffer_s[i][mpi_diag_correction_offset + j] += values[k];
              } else {
                // mpi_ext_ext_cols_s[i][ext_ext_ptr] = col_lid;
                // mpi_ext_ext_values_s[i][ext_ext_ptr] = values[k];
                ibuffer_s[i][3 * (send_size + 1) + ext_ext_offset + ext_ext_ptr] =
                    col_lid;
                rbuffer_s[i][ext_ext_offset + ext_ext_ptr] = values[k];
                ++ext_ext_ptr;
              }
            }
          } else {
            if (col >= ids_offset_r[i_recv_neighb]
                && col < ids_offset_r[i_recv_neighb + 1]) {
              // CHECK IF BELONG TO DOMAIN I

              // mpi_ext_int_cols_s[i][ext_int_ptr] = col -ids_offset_r[i_recv_neighb]  ;
              // mpi_ext_int_values_s[i][ext_int_ptr] = values[k];

              ibuffer_s[i][3 * (send_size + 1) + ext_int_offset + ext_int_ptr] =
                  col - ids_offset_r[i_recv_neighb];
              rbuffer_s[i][ext_int_offset + ext_int_ptr] = values[k];
              ++ext_int_ptr;
            } else {
              // BELONG TO DOMAIN J=recv_neighb_id
              int recv_neighb_id = recv_info.getNeighbId(col);
              // mpi_ext_ext2_cols_s[i][2*ext_ext2_ptr]   = recv_info.m_uids[col
              // -ids_offset_r[0]] ;
              // mpi_ext_ext2_cols_s[i][2*ext_ext2_ptr+1] =
              // recv_info.m_ranks[recv_neighb_id] ;
              // mpi_ext_ext2_values_s[i][ext_ext2_ptr]   = values[k];

              ibuffer_s[i][3 * (send_size + 1) + ext_ext2_offset + 2 * ext_ext2_ptr] =
                  recv_info.m_uids[col - ids_offset_r[0]];
              ibuffer_s[i][3 * (send_size + 1) + ext_ext2_offset + 2 * ext_ext2_ptr + 1] =
                  recv_info.m_ranks[recv_neighb_id];
              rbuffer_s[i][ext_ext2_offset + ext_ext2_ptr] = values[k];
              ++ext_ext2_ptr;
            }
          }
        }
      }
      assert(ext_int_ptr == nnz_ext_int_send[i]);
      ibuffer_s[i][send_size] = ext_int_ptr;

      assert(ext_ext_ptr == nnz_ext_ext_send[i]);
      ibuffer_s[i][2 * send_size + 1] = ext_ext_ptr;

      assert(ext_ext2_ptr == nnz_ext_ext2_send[i]);
      ibuffer_s[i][2 * (send_size + 1) + send_size] = ext_ext2_ptr;

      send_request0[i] =
          Arccore::MessagePassing::mpSend(m_parallel_mng, ibuffer_s[i], ranki, false);
      send_request1[i] =
          Arccore::MessagePassing::mpSend(m_parallel_mng, rbuffer_s[i], ranki, false);
    }
  }

  UniqueArray<MessagePassing::Request> recv_request1(num_neighbours_r);
  UniqueArray<UniqueArray<Integer>> ibuffer_r(num_neighbours_r);
  UniqueArray<UniqueArray<ValueT>> rbuffer_r(num_neighbours_r);

  for (std::size_t i = 0; i < num_neighbours_r; ++i) {
    int ranki = dist_info.m_recv_info.m_ranks[i];

    nrows[i] = count_recv[4 * i];
    nnz_ext_int[i] = count_recv[4 * i + 1];
    nnz_ext_ext[i] = count_recv[4 * i + 2];
    nnz_ext_ext2[i] = count_recv[4 * i + 3];

    ibuffer_r[i].resize(
        3 * (nrows[i] + 1) + nnz_ext_int[i] + nnz_ext_ext[i] + 2 * nnz_ext_ext2[i]);
    rbuffer_r[i].resize(nnz_ext_int[i] + nnz_ext_ext[i] + nnz_ext_ext2[i] + nrows[i]);

    recv_request0[i] =
        Arccore::MessagePassing::mpReceive(m_parallel_mng, ibuffer_r[i], ranki, false);
    recv_request1[i] =
        Arccore::MessagePassing::mpReceive(m_parallel_mng, rbuffer_r[i], ranki, false);
  }
  Arccore::MessagePassing::mpWaitAll(m_parallel_mng, send_request0);
  Arccore::MessagePassing::mpWaitAll(m_parallel_mng, send_request1);
  Arccore::MessagePassing::mpWaitAll(m_parallel_mng, recv_request0);
  Arccore::MessagePassing::mpWaitAll(m_parallel_mng, recv_request1);

  Integer domain_offset = A.getLocalOffset();

  m_ghost_nrows = dist_info.m_ghost_nrow;
  m_ghost_nnz = 0;
  for (std::size_t i = 0; i < num_neighbours_r; ++i) {
    m_ghost_nnz += ibuffer_r[i][nrows[i]] + ibuffer_r[i][nrows[i] + 1 + nrows[i]]
        + ibuffer_r[i][2 * (nrows[i] + 1) + nrows[i]];
  }

  m_ndofs = m_local_nrows + m_ghost_nrows;
  m_nnz = m_local_nnz + m_ghost_nnz;

  {
    auto& profile_dirichlet = m_csr_matrix_dirichlet.internal().getCSRProfile();
    profile_dirichlet.init(m_ndofs, m_nnz);
    m_csr_matrix_dirichlet.allocate();

    auto& profile_neumann = m_csr_matrix_neumann.internal().getCSRProfile();
    profile_neumann.init(m_ndofs, m_nnz);
    m_csr_matrix_neumann.allocate();

    m_diag_correction.resize(m_ghost_nrows);

    auto kcol_dirichlet = profile_dirichlet.kcol();
    auto cols_dirichlet = profile_dirichlet.cols();
    ValueT* values_dirichlet = m_csr_matrix_dirichlet.getAddressData();

    auto kcol_neumann = profile_neumann.kcol();
    auto cols_neumann = profile_neumann.cols();
    ValueT* values_neumann = m_csr_matrix_neumann.getAddressData();

    int offset = 0;
    for (int irow = 0; irow < m_local_nrows; ++irow) {
      kcol_dirichlet[irow] = offset;
      kcol_neumann[irow] = offset;
      for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {
        cols_dirichlet[k] = cols[k];
        values_dirichlet[k] = values[k];
        cols_neumann[k] = cols[k];
        values_neumann[k] = values[k];
      }
      offset += kcol[irow + 1] - kcol[irow];
    }

    int irow_ext = 0;
    for (std::size_t i = 0; i < num_neighbours_r; ++i) {
      ConstArrayView<Integer> mpi_ext_int_kcol(nrows[i] + 1, &ibuffer_r[i][0]);
      ConstArrayView<Integer> mpi_ext_ext_kcol(nrows[i] + 1, &ibuffer_r[i][nrows[i] + 1]);
      ConstArrayView<Integer> mpi_ext_ext2_kcol(
          nrows[i] + 1, &ibuffer_r[i][2 * (nrows[i] + 1)]);

      int mpi_ext_int_nnz = mpi_ext_int_kcol[nrows[i]];
      int mpi_ext_int_offset = 0;
      ConstArrayView<Integer> mpi_ext_int_cols(
          mpi_ext_int_nnz, &ibuffer_r[i][3 * (nrows[i] + 1) + mpi_ext_int_offset]);
      ConstArrayView<ValueT> mpi_ext_int_values(
          mpi_ext_int_nnz, &rbuffer_r[i][mpi_ext_int_offset]);

      int mpi_ext_ext_nnz = mpi_ext_ext_kcol[nrows[i]];
      int mpi_ext_ext_offset = mpi_ext_int_offset + mpi_ext_int_nnz;
      ConstArrayView<Integer> mpi_ext_ext_cols(
          mpi_ext_ext_nnz, &ibuffer_r[i][3 * (nrows[i] + 1) + mpi_ext_ext_offset]);
      ConstArrayView<ValueT> mpi_ext_ext_values(
          mpi_ext_ext_nnz, &rbuffer_r[i][mpi_ext_ext_offset]);

      int mpi_ext_ext2_nnz = mpi_ext_ext2_kcol[nrows[i]];
      int mpi_ext_ext2_offset = mpi_ext_ext_offset + mpi_ext_ext_nnz;
      ConstArrayView<Integer> mpi_ext_ext2_cols(
          2 * mpi_ext_ext2_nnz, &ibuffer_r[i][3 * (nrows[i] + 1) + mpi_ext_ext2_offset]);
      ConstArrayView<ValueT> mpi_ext_ext2_values(
          mpi_ext_ext2_nnz, &rbuffer_r[i][mpi_ext_ext2_offset]);

      int diag_correction_offset = mpi_ext_ext2_offset + mpi_ext_ext2_nnz;
      ConstArrayView<ValueT> mpi_diag_correction(
          nrows[i], &rbuffer_r[i][diag_correction_offset]);

      std::size_t id_offset_r = ids_offset_r[i];
      for (int irow = 0; irow < nrows[i]; ++irow) {
        kcol_dirichlet[m_local_nrows + irow_ext] = offset;
        kcol_neumann[m_local_nrows + irow_ext] = offset;

        m_diag_correction[irow_ext] = mpi_diag_correction[irow];
        for (int k = mpi_ext_int_kcol[irow]; k < mpi_ext_int_kcol[irow + 1]; ++k) {
          int col_lid = ids_s[mpi_ext_int_cols[k] + ids_offset_s[i]];
          // int col_uid = domain_offset + col_lid ;

          cols_dirichlet[offset] = col_lid;
          cols_neumann[offset] = col_lid;

          values_dirichlet[offset] = mpi_ext_int_values[k];
          values_neumann[offset] = mpi_ext_int_values[k];

          ++offset;
        }
        for (int k = mpi_ext_ext2_kcol[irow]; k < mpi_ext_ext2_kcol[irow + 1]; ++k) {
          int col_uid = mpi_ext_ext2_cols[2 * k];
          int col_domk = mpi_ext_ext2_cols[2 * k + 1];
          int neighb_id = recv_info.getRankNeighbId(col_domk);
          int col_lid =
              neighb_id == -1 ? -1 : recv_info.getLocalIdFromUid(neighb_id, col_uid);
          // int glid    = col_lid==-1 ? -1 : col_lid-m_local_nrows ;
          if (col_lid != -1) {
            // int col_uid            = uids_r[glid] ;
            cols_dirichlet[offset] = col_lid;
            cols_neumann[offset] = col_lid;

            values_dirichlet[offset] = mpi_ext_ext2_values[k];
            values_neumann[offset] = mpi_ext_ext2_values[k];
            ++offset;
          } else {
            m_diag_correction[irow_ext] += mpi_ext_ext2_values[k];
          }
        }
        for (int k = mpi_ext_ext_kcol[irow]; k < mpi_ext_ext_kcol[irow + 1]; ++k) {
          int col_lid = mpi_ext_ext_cols[k] + id_offset_r;
          // int glid = col_lid-m_local_nrows ;
          // int col_uid = uids_r[glid] ;
          cols_dirichlet[offset] = col_lid;
          cols_neumann[offset] = col_lid;

          values_dirichlet[offset] = mpi_ext_ext_values[k];
          values_neumann[offset] = mpi_ext_ext_values[k];

          if (col_lid == m_local_nrows + irow_ext)
            values_neumann[offset] += m_diag_correction[irow_ext];

          ++offset;
        }

        ++irow_ext;
      }
    }
    assert(irow_ext == m_ghost_nrows);
    kcol_dirichlet[m_local_nrows + irow_ext] = offset;
    kcol_neumann[m_local_nrows + irow_ext] = offset;
#ifdef DEBUG
    {
      std::stringstream filename;
      filename << "DirichletMatrix" << m_parallel_mng->commRank() << ".txt";
      std::ofstream fout(filename.str());
      for (int irow = 0; irow < m_ndofs; ++irow) {
        fout << irow << ":";
        for (int k = kcol_dirichlet[irow]; k < kcol_dirichlet[irow + 1]; ++k) {
          fout << "(" << cols_dirichlet[k] << "," << values_dirichlet[k] << ");";
        }
        fout << std::endl;
      }
    }
    {
      std::stringstream filename;
      filename << "NeumannMatrix" << m_parallel_mng->commRank() << ".txt";
      std::ofstream fout(filename.str());
      for (int irow = 0; irow < m_ndofs; ++irow) {
        fout << irow << ":";
        for (int k = kcol_neumann[irow]; k < kcol_neumann[irow + 1]; ++k) {
          fout << "(" << cols_neumann[k] << "," << values_neumann[k] << ");";
        }
        fout << std::endl;
      }
    }
#endif
  }
}

template <typename ValueT>
HPDDM::MatrixCSR<ValueT>*
MatrixInternal<ValueT>::_createDirchletMatrix(
    typename MatrixInternal<ValueT>::CSRMatrixType const& A)
{

  _computeMPIGhostMatrix(A);

  auto& profile_dirichlet = m_csr_matrix_dirichlet.internal().getCSRProfile();
  auto kcol = profile_dirichlet.kcol();
  auto cols = profile_dirichlet.cols();
  ValueT* values = m_csr_matrix_dirichlet.getAddressData();

  return new HPDDM::MatrixCSR<ValueT>(m_ndofs, m_ndofs, m_nnz, values, kcol, cols,
      /*symetric        */ false,
      /*take ownner ship*/ false);
}

template <typename ValueT>
HPDDM::MatrixCSR<ValueT>*
MatrixInternal<ValueT>::_createNeunmanMatrix(
    typename MatrixInternal<ValueT>::CSRMatrixType const& A)
{
  auto& profile_neumann = m_csr_matrix_neumann.internal().getCSRProfile();
  auto kcol = profile_neumann.kcol();
  auto cols = profile_neumann.cols();
  ValueT* values = m_csr_matrix_neumann.getAddressData();

  return new HPDDM::MatrixCSR<ValueT>(m_ndofs, m_ndofs, m_nnz, values, kcol, cols,
      /*symetric        */ false,
      /*take ownner ship*/ false);
}

template <typename ValueT>
void
MatrixInternal<ValueT>::_computeUnitPartition()
{
  int k = 0;
  m_unit_partition.resize(m_ndofs);
  std::fill(m_unit_partition.begin(), m_unit_partition.begin() + m_local_nrows, 1.);
  std::fill(m_unit_partition.begin() + m_local_nrows, m_unit_partition.end(), 0.);
}

template <typename ValueT>
void
MatrixInternal<ValueT>::_computeOverlapConnectivity(const CSRMatrixType& A)
{
  using namespace Arccore;
  auto const& dist_info = A.getDistStructInfo();
  Integer domain_offset = A.getLocalOffset();

  auto const& send_info = dist_info.m_send_info;
  std::size_t num_neighbours_s = dist_info.m_send_info.m_num_neighbours;
  auto const& ids_offset_s = dist_info.m_send_info.m_ids_offset;
  auto const& ids_s = dist_info.m_send_info.m_ids;

  auto const& recv_info = dist_info.m_recv_info;
  std::size_t num_neighbours_r = dist_info.m_recv_info.m_num_neighbours;
  auto const& ids_offset_r = dist_info.m_recv_info.m_ids_offset;
  auto const& uids_r = dist_info.m_recv_info.m_uids;

  for (int ineighb = 0; ineighb < num_neighbours_r; ++ineighb) {
    std::map<int, int> overlap_uid_lids;
    m_overlap.push_back(recv_info.m_ranks[ineighb]);
    int ineighb_s = send_info.getRankNeighbId(recv_info.m_ranks[ineighb]);
    if (ineighb_s != -1) {
      for (int k = ids_offset_s[ineighb_s]; k < ids_offset_s[ineighb_s + 1]; ++k) {
        overlap_uid_lids[domain_offset + ids_s[k]] = ids_s[k];
      }
    }
    for (int k = ids_offset_r[ineighb]; k < ids_offset_r[ineighb + 1]; ++k) {
      overlap_uid_lids[uids_r[k - ids_offset_r[0]]] = k;
    }
    std::size_t intersect_size = overlap_uid_lids.size();
    std::vector<int> intersect;
    intersect.reserve(intersect_size);
    for (auto const& iter : overlap_uid_lids) {
      intersect.push_back(iter.second);
    }
    m_mapping.push_back(std::move(intersect));
  }
}

template <typename ValueT>
void
MatrixInternal<ValueT>::_compute(HPDDM::MatrixCSR<ValueT>* matrix_dirichlet,
    HPDDM::MatrixCSR<ValueT>* matrix_neumann, unsigned short nu,
    bool schwarz_coarse_correction)
{
  assert(m_parallel_mng != nullptr);
  int rank = m_parallel_mng->commRank();
  if (m_parallel_mng->commSize() > 1) {
    auto* pm =
        dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(m_parallel_mng);
    assert(pm);
    MPI_Comm const* comm = static_cast<MPI_Comm const*>(pm->getMPIComm());
    // m_hpddm_matrix.Subdomain::initialize(&DiriCSR, m_overlap, m_mapping);
    m_matrix.HPDDM::Subdomain<ValueT>::initialize(
        matrix_dirichlet, m_overlap, m_mapping, const_cast<MPI_Comm*>(comm));
    // m_matrix.Subdomain::initialize(&m_matrix_dirichlet, m_overlap, m_mapping);
    decltype(m_mapping)().swap(m_mapping);
    m_matrix.multiplicityScaling(m_unit_partition.data());
    m_matrix.initialize(m_unit_partition.data());
    if (schwarz_coarse_correction) {
      if (rank == 0)
        std::cout << "HPDDM Using Two Levels" << m_ndofs << " " << m_nnz << std::endl;
      // double& ref = opt["geneo_nu"];
      // unsigned short nu = ref;
      Real eigen_solver_time = 0.;
      if (nu > 0) {
        Alien::BaseSolverStater::Sentry s(eigen_solver_time);
        m_matrix.template solveGEVP<EIGENSOLVER>(matrix_neumann);
        // nu = opt["geneo_nu"];
        // m_hpddm_matrix.super::initialize(nu);
        m_matrix.HPDDMMatrixType::super::initialize(nu);
        m_matrix.buildTwo(*comm);
      } else
        std::cout << "Warning ! Case nu = 0 not supported. Falling back to 1 level method"
                  << std::endl;
      if (rank == 0)
        std::cout << "EIGEN SOLVER TIME : " << eigen_solver_time << std::endl;
    }
    m_matrix.callNumfact();
  }
}

template <typename ValueT>
void
MatrixInternal<ValueT>::compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& A,
    unsigned short nu, bool schwarz_coarse_correction)
{
  m_parallel_mng = parallel_mng;
  m_local_nrows = A.getLocalSize();
  m_local_nnz = A.getProfile().getNnz();
  m_ghost_nrows = A.getGhostSize();

  HPDDM::MatrixCSR<ValueT>* matrix_dirichlet = _createDirchletMatrix(A);
  HPDDM::MatrixCSR<ValueT>* matrix_neumann = _createNeunmanMatrix(A);
  _computeUnitPartition();
  _computeOverlapConnectivity(A);
  _compute(matrix_dirichlet, matrix_neumann, nu, schwarz_coarse_correction);
}

template <typename ValueT>
void
MatrixInternal<ValueT>::compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& Ad,
    const CSRMatrixType& An, unsigned short nu, bool schwarz_coarse_correction)
{
  m_parallel_mng = parallel_mng;
  m_local_nrows = Ad.getLocalSize();
  m_local_nnz = Ad.getProfile().getNnz();

  HPDDM::MatrixCSR<ValueT>* matrix_dirichlet = _createDirchletMatrix(Ad);
  HPDDM::MatrixCSR<ValueT>* matrix_neumann = _createNeunmanMatrix(An);
  _computeUnitPartition();
  _computeOverlapConnectivity(An);

  _compute(matrix_dirichlet, matrix_neumann, nu, schwarz_coarse_correction);
}

template class MatrixInternal<double>;

END_HPDDMINTERNAL_NAMESPACE

namespace Alien {
/*---------------------------------------------------------------------------*/
template <typename ValueT>
HPDDMMatrix<ValueT>::HPDDMMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::hpddm>::name())
{
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();

  if (row_space.size() != col_space.size())
    throw FatalErrorException("HPDDM matrix must be square");
}

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

template class HPDDMMatrix<double>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
