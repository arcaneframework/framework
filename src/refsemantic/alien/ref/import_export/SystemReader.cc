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

#include <alien/ref/import_export/SystemReader.h>

#include <string>
#include <vector>

#include <alien/ref/AlienRefSemantic.h>

#ifdef ALIEN_USE_HDF5
#include <hdf5.h>
#endif

#ifdef ALIEN_USE_LIBXML2
#include <libxml/parser.h>
#endif

#include "HDF5Tools.h"

namespace Alien
{

SystemReader::SystemReader(
std::string const& filename, std::string format, IMessagePassingMng* parallel_mng)
: m_filename(filename)
, m_format(format)
, m_prec(6)
, m_parallel_mng(parallel_mng)
{
  m_rank = 0;
  m_nproc = 1;
  if (m_parallel_mng) {
    m_parallel_mng->commRank();
    m_parallel_mng->commSize();
  }
}

SystemReader::~SystemReader() {}

template <typename FileNodeT>
void SystemReader::_readMatrixInfo(Importer& importer, FileNodeT& parent_node, int& nrows,
                                   int& ncols, int& nnz, int& blk_size, int& blk_size2)
{
  FileNodeT matrix_info_node = importer.openFileNode(parent_node, "matrix-info");
  importer.read(matrix_info_node, "nrows", nrows);
  importer.read(matrix_info_node, "ncols", ncols);
  importer.read(matrix_info_node, "nnz", nnz);
  {
    std::vector<int>& i32buffer = importer.i32buffer;
    i32buffer.resize(2);
    importer.read(matrix_info_node, "blk-size", i32buffer);
    blk_size = i32buffer[0];
    blk_size2 = i32buffer[1];
  }
  importer.closeFileNode(matrix_info_node);
}

template <typename FileNodeT>
void SystemReader::_readCSRProfile(Importer& importer, FileNodeT& parent_node, int& nrows,
                                   int& nnz, std::vector<int>& kcol, std::vector<int>& cols)
{
  FileNodeT profile_node = importer.openFileNode(parent_node, "profile");

  importer.read(profile_node, "nrows", nrows);
  importer.read(profile_node, "nnz", nnz);

  kcol.resize(nrows + 1);
  importer.read(profile_node, "kcol", kcol);

  cols.resize(nnz);
  importer.read(profile_node, "cols", cols);

  importer.closeFileNode(profile_node);
}

template <typename FileNodeT>
void SystemReader::_readMatrixValues(Importer& importer, FileNodeT& parent_node, int& size,
                                     int& blk_size, int& blk_size2, std::vector<double>& values)
{
  std::vector<int>& i32buffer = importer.i32buffer;
  FileNodeT data_node = importer.openFileNode(parent_node, "data");
  {
    {
      FileNodeT info_node = importer.openFileNode(data_node, "struct-info");

      importer.read(info_node, "size", size);

      {
        i32buffer.reserve(2);
        importer.read(info_node, "blk-size", i32buffer);
        blk_size = i32buffer[0];
        blk_size = i32buffer[1];
      }
      importer.closeFileNode(info_node);
    }
    values.resize(size * blk_size * blk_size2);
    importer.read(data_node, "values", values);
  }
  importer.closeFileNode(data_node);
}

void SystemReader::read(Matrix& A)
{

  typedef Importer::FileNode FileNode;

  Importer importer(m_filename, m_format, m_prec);

  int nrows = 0;
  int ncols = 0;
  int nnz = 0;
  int blk_size = 1;
  std::vector<int> kcol;
  std::vector<int> cols;
  std::vector<double> values;

  FileNode root_node = importer.openFileNode("system");
  {
    FileNode matrix_node = importer.openFileNode(root_node, "matrix");
    {
      int file_blk_size;
      int file_blk_size2;
      _readMatrixInfo(
      importer, matrix_node, nrows, ncols, nnz, file_blk_size, file_blk_size2);
      if (file_blk_size != blk_size || file_blk_size2 != blk_size)
        throw FatalErrorException(
        A_FUNCINFO, "Incompatible  block size with imported system");

      const auto& dist = A.distribution();
      int offset = dist.rowOffset();
      int lsize = dist.localRowSize();

      if (offset + lsize > nrows)
        throw FatalErrorException(
        A_FUNCINFO, "Incompatible  space size with imported system");

      int file_nrows;
      int file_nnz;

      // TODO: only read profile from offset to offset + lsize
      //       or explicitly distribute profile according to distribution
      _readCSRProfile(importer, matrix_node, file_nrows, file_nnz, kcol, cols);

      if (file_nrows != nrows || file_nnz != nnz)
        throw FatalErrorException(A_FUNCINFO, "Incoherent matrix profile");

      {
        Alien::MatrixProfiler profiler(A);

        for (int irow = offset; irow < offset + lsize; ++irow) {
          for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {
            profiler.addMatrixEntry(irow, cols[k]);
          }
        }
      }

      // TODO: only read values from offset to offset + lsize
      //       or explicitly distribute values according to distribution

      _readMatrixValues(
      importer, matrix_node, file_nnz, file_blk_size, file_blk_size2, values);

      if (file_nnz != nnz || file_blk_size != 1 || file_blk_size2 != 1)
        throw FatalErrorException(A_FUNCINFO, "Incoherent matrix values");

      Alien::Common::ProfiledMatrixBuilder builder(
      A, Alien::ProfiledMatrixOptions::eResetValues);

      for (int irow = offset; irow < offset + lsize; ++irow) {
        for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {
          builder(irow, cols[k]) = values[k];
        }
      }
    }
    importer.closeFileNode(matrix_node);
  }
  importer.closeFileNode(root_node);
}

void SystemReader::read(BlockMatrix& A)
{
  typedef Importer::FileNode FileNode;

  Importer importer(m_filename, m_format, m_prec);

  int nrows = 0;
  int ncols = 0;
  int nnz = 0;
  int blk_size = 1;
  int blk_size2 = 1;
  std::vector<int> kcol;
  std::vector<int> cols;
  std::vector<double> values;

  FileNode root_node = importer.openFileNode("system");
  {
    FileNode matrix_node = importer.openFileNode(root_node, "matrix");
    {
      int file_blk_size;
      int file_blk_size2;
      _readMatrixInfo(
      importer, matrix_node, nrows, ncols, nnz, file_blk_size, file_blk_size2);

      if (file_blk_size != file_blk_size2)
        throw FatalErrorException(A_FUNCINFO, "Non square block not supported");

      blk_size = file_blk_size;
      blk_size2 = blk_size;
      const Alien::Block block(blk_size);

      A = BlockMatrix(nrows, nrows, block, m_parallel_mng);

      const auto& dist = A.distribution();

      int offset = dist.rowOffset();
      int lsize = dist.localRowSize();

      if (offset + lsize > nrows)
        throw FatalErrorException(
        A_FUNCINFO, "Incompatible  space size with imported system");

      int file_nrows;
      int file_nnz;
      // TODO: only read profile from offset to offset + lsize
      //       or explicitly distribute profile according to distribution
      _readCSRProfile(importer, matrix_node, file_nrows, file_nnz, kcol, cols);

      if (file_nrows != nrows || file_nnz != nnz)
        throw FatalErrorException(A_FUNCINFO, "Incoherent matrix profile");

      {
        Alien::MatrixProfiler profiler(A);

        for (int irow = offset; irow < offset + lsize; ++irow) {
          for (int k = kcol[irow]; k < kcol[irow + 1]; ++k)
            profiler.addMatrixEntry(irow, cols[k]);
        }
      }

      // TODO: only read values from offset to offset + lsize
      //       or explicitly distribute values according to distribution
      _readMatrixValues(
      importer, matrix_node, file_nnz, file_blk_size, file_blk_size2, values);

      if (file_nnz != nnz || file_blk_size != blk_size || file_blk_size2 != blk_size2)
        throw FatalErrorException(A_FUNCINFO, "Incoherent matrix values");

      {
        Alien::ProfiledBlockMatrixBuilder builder(
        A, Alien::ProfiledBlockMatrixBuilderOptions::eResetValues);

        //        dist = A.distribution();
        //        offset = dist.rowOffset();
        //        lsize = dist.localRowSize();

        for (int irow = offset; irow < offset + lsize; ++irow) {
          for (int k = kcol[irow]; k < kcol[irow + 1]; ++k)
            builder(irow, cols[k]) =
            Array2View<double>(&values[k * blk_size * blk_size], blk_size, blk_size);
        }
      }
    }
    importer.closeFileNode(matrix_node);
  }
  importer.closeFileNode(root_node);
}
/*
  void SystemReader::read(Vector & x)
  {
    typedef Importer::FileNode FileNode ;

    Importer importer(m_filename,m_format,m_prec) ;
    std::vector<double>& rbuffer  = importer.rbuffer ;
    std::vector<int>& i32buffer   = importer.i32buffer ;

    int nrows = 0 ;
    int blk_size = 1 ;

    FileNode root_node = importer.openFileRootNode() ;
    {
      FileNode vector_node = importer.openFileNode(root_node,"vector") ;
      {
        FileNode info_node = importer.openFileNode(vector_node,"struct-info") ;
        {
          FileNode nrows_node = importer.openFileNode(info_node,"nrows") ;
          i32buffer.resize(1) ;
          importer.read(nrows_node,i32buffer) ;
          importer.closeFileNode(nrows_node) ;
          nrows = i32buffer[0] ;
        }
        {
          FileNode blk_size_node = importer.openFileNode(info_node,"blk-size") ;
          i32buffer.resize(1) ;
          importer.read(blk_size_node,i32buffer) ;
          importer.closeFileNode(blk_size_node) ;
          blk_size = i32buffer[0] ;
        }
        importer.closeFileNode(info_node) ;

        x = Vector(nrows,m_parallel_mng) ;

        const auto& dist = x.distribution();
        int offset = dist.offset();
        int lsize = dist.localSize();

        FileNode data_node = importer.openFileNode(vector_node,"data") ;
        {


          FileNode values_node = importer.openFileNode(data_node,"values") ;
          rbuffer.resize(nrows*blk_size) ;
          importer.read(values_node,rbuffer) ;
          importer.closeFileNode(values_node) ;

          // Builder du vecteur
          Alien::VectorWriter writer(x);

          // On remplit le vecteur
          for(int i = 0; i < lsize; ++i) {
            writer[i+offset] = rbuffer[(offset+i)*blk_size];
          }
        }
        importer.closeFileNode(data_node) ;
      }
    importer.closeFileNode(vector_node) ;
   }
   importer.closeFileNode(system_node) ;
   importer.closeFileRootNode(root_node) ;
  }



  void SystemReader::read(BlockVector & x)
  {

    typedef Exporter::FileNode FileNode ;

    Importer importer(m_filename,m_format,m_prec) ;
    std::vector<double>& rbuffer  = importer.rbuffer ;
    std::vector<int>& i32buffer   = importer.i32buffer ;

    int nrows = 0 ;
    int blk_size = 1 ;

    FileNode root_node = importer.openFileRootNode() ;
    FileNode system_node = importer.openFileNode(root_node,"system") ;
    {
      FileNode vector_node = importer.openFileNode(root_node,"vector") ;
      {
        FileNode info_node = importer.openFileNode(vector_node,"struct-info") ;
        {
          FileNode nrows_node = importer.openFileNode(info_node,"nrows") ;
          i32buffer.resize(1) ;
          importer.read(nrows_node,i32buffer) ;
          importer.closeFileNode(nrows_node) ;
          nrows = i32buffer[0] ;
        }
        {
          FileNode blk_size_node = importer.openFileNode(info_node,"blk-size") ;
          i32buffer.resize(1) ;
          importer.read(blk_size_node,i32buffer) ;
          importer.closeFileNode(blk_size_node) ;
          blk_size = i32buffer[0] ;
        }
        importer.closeFileNode(info_node) ;

        const Alien::Block block(blk_size);
        x = BlockVector(nrows,block,m_parallel_mng) ;

        const auto& dist = x.distribution();
        int offset = dist.offset();
        int lsize = dist.localSize();

        FileNode data_node = importer.openFileNode(vector_node,"data") ;
        {
          FileNode values_node = importer.openFileNode(data_node,"values") ;
          rbuffer.resize(nrows*blk_size) ;
          importer.read(values_node,rbuffer) ;
          importer.closeFileNode(values_node) ;

          // Builder du vecteur
          Alien::BlockVectorWriter writer(x);

          // On remplit le vecteur
          for(int i = 0; i < lsize; ++i) {
            writer[i+offset] = ArrayView<double>(blk_size,&rbuffer[(offset+i)*blk_size]);
          }
        }
        importer.closeFileNode(data_node) ;
      }
    importer.closeFileNode(vector_node) ;
   }
   importer.closeFileNode(system_node) ;
   importer.closeFileRootNode(root_node) ;
  }
  */
} /* namespace Alien */
