﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "IFPMatrix.h"
#include <set>
#include <vector>

#include <IFPSolver.h>
#include <alien/kernels/ifp/data_structure/IFPVector.h>
#include <alien/kernels/ifp/data_structure/IFPSolverInternal.h>
#include <alien/kernels/ifp/IFPSolverBackEnd.h>
#include <alien/core/impl/MultiMatrixImpl.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

IFPMatrix::IFPMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::ifpsolver>::name())
, m_internal(NULL)
, m_node_list_ref(NULL)
, m_row_offset(0)
, m_graph_initialized(false)
, m_symmetric_profile(true)
, m_sum_first_eq(1)
, m_space0(NULL)
, m_space1(NULL)
{
  m_row_offset = multi_impl->distribution().rowOffset();
}

/*---------------------------------------------------------------------------*/

IFPMatrix::~IFPMatrix()
{
  delete m_internal;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::initMatrix(Integer links_num, // num of links
    Integer global_nodes_num, // global num of nodes
    Integer nodes_num, // num of nodes [includes ghost nodes]
    Integer local_nodes_num, // num of nodes in linear system [without ghost nodes]
    Integer max_node_id,
    Integer equations_num, // num of equations by link
    Integer unknowns_num, // num of unknowns by link
    Integer* nodeList, // list of nodes as local Arcane num [size=nodes_num]
    Integer* globalNodeList, // list of nodes as global Arcane num [size=nodes_num]
    Integer* nodeToLocalRow, Integer* rowUidList,
    Integer* i_node, // list of nodes i of links (ij) [local index; size=links_num]
    Integer* global_i_node, // idem but global index
    Integer* j_node, // list of nodes i of links (ij)
    Integer* global_j_node // idem but global index
    )
{
  /* m_node_to_local_row describes local nodes only.
   * Ghost nodes verifies m_node_to_local_row[node] == -1
   * Then, on the local matrix, for each link, only well defined
   * nodes (m_node_to_local_row != -1) is used as row index.
   * If a link(i,j) has a ghost node j, it will be duplicated as link(i,j)
   * on owner(j) with with i as a ghost node.
   */
  if (MatrixInternal::isInstancied()) {
    if (MatrixInternal::multiImpl() == m_multi_impl) {
      // space().message() << "Free IFPSolver owned internal";
      delete m_internal;
    } else {
      // space().message() << "Free IFPSolver other internal";
      m_multi_impl->release<BackEnd::tag::ifpsolver>();
    }
  }
  alien_debug([&] {cout() << " NEW IFP INTERNAL";});

  // Check non used internal
  ALIEN_ASSERT((!MatrixInternal::isInstancied()),
      ("Inconsistent error : IFPSolver Matrix internal is still alive"));

  // New internal
  m_internal = new MatrixInternal(m_multi_impl);

  m_node_list_ref = nodeList;

  /*
   * compute the size of matrix row :
   * for one link (ij), there is one non zero entry on row(i) and one on row(j)
   * be careful store row size on (node+1)
   * row_size is allocated as fortran array, starting at index 1
   */
  Integer* row_size = new Integer[local_nodes_num + 1];

  for (Integer node = 0; node <= local_nodes_num; ++node) {
    row_size[node] = 0;
  }

  for (Integer link = 0; link < links_num; ++link) {
    const Integer rowi = nodeToLocalRow[i_node[link]];

    if (rowi != -1) {
      row_size[rowi + 1]++;
    }

    const Integer rowj = nodeToLocalRow[j_node[link]];

    if (rowj != -1) {
      row_size[rowj + 1]++;
    }
  }

  // Compte Column Index Pointer for CSR Format
  Integer* columnIndexPtr = new Integer[local_nodes_num + 1];

  columnIndexPtr[0] = 0;
  for (Integer node = 1; node < local_nodes_num; ++node) {
    columnIndexPtr[node] = columnIndexPtr[node - 1] + row_size[node];
    row_size[node] = 0;
  }
  columnIndexPtr[local_nodes_num] =
      columnIndexPtr[local_nodes_num - 1] + row_size[local_nodes_num];

  // Fill ColumnIndex Array
  Integer columnIndexSize = columnIndexPtr[local_nodes_num];

  Integer* columnIndex = new Integer[columnIndexSize];

  // Loop in Link (ij) : on row i, ColIndew=j and on row j ColIndex=i
  for (Integer link = 0; link < links_num; ++link) {
    const Integer rowi = nodeToLocalRow[i_node[link]];
    if (rowi != -1) {
      columnIndex[columnIndexPtr[rowi] + row_size[rowi]] = global_j_node[link];
      row_size[rowi]++;
    }

    const Integer rowj = nodeToLocalRow[j_node[link]];
    if (rowj != -1) {
      columnIndex[columnIndexPtr[rowj] + row_size[rowj]] = global_i_node[link];
      row_size[rowj]++;
    }
  }
  // IFP Solver init
  F2C(ifpsolverinitgraph)
  (&global_nodes_num, &local_nodes_num, &nodes_num, &max_node_id, &links_num,
      &columnIndexSize, columnIndex, columnIndexPtr, nodeList, rowUidList, i_node, j_node,
      global_i_node, global_j_node, &equations_num, &unknowns_num);
  delete[] row_size;
  delete[] columnIndex;
  delete[] columnIndexPtr;

  m_graph_initialized = true;

  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::initMatrix(Integer links_num, // num of links
    Integer global_nodes_num, // global num of nodes
    Integer nodes_num, // num of nodes [includes ghost nodes]
    Integer local_nodes_num, // num of nodes in linear system [without ghost nodes]
    Integer max_node_id,
    Integer equations_num, // num of equations by link
    Integer unknowns_num, // num of unknowns by link
    Integer* nodeList, // list of nodes as local Arcane num [size=nodes_num]
    Integer* globalNodeList, // list of nodes as global Arcane num [size=nodes_num]
    Integer* nodeToLocalRow, Integer* rowUidList,
    Integer* i_node, // list of nodes i of links (ij) [local index; size=links_num]
    Integer* global_i_node, //    idem but global index
    Integer* j_node, // list of nodes i of links (ij)
    Integer* global_j_node, // idem but global index
    Integer* ass_elem_node_ptr, // List of offset of Associated element List
    Integer* ass_elem_node, // List of Associated elements of links
    Integer* global_ass_elem_node, // idem with global index
    Integer extra_eq_num, Integer global_extra_eq_num, Integer* extra_eq_ids,
    Integer* extra_eq_elem_node_ptr, Integer* extra_eq_elem_node)
{
  /* m_node_to_local_row describes local nodes only.
   * Ghost nodes verifies m_node_to_local_row[node] == -1
   * Then, on the local matrix, for each link, only well defined
   * nodes (m_node_to_local_row != -1) is used as row index.
   * If a link(i,j) has a ghost node j, it will be duplicated as link(i,j)
   * on owner(j) with with i as a ghost node.
   */
  if (MatrixInternal::isInstancied()) {
    if (MatrixInternal::multiImpl() == m_multi_impl) {
      // space().message() << "Free IFPSolver owned internal";
      delete m_internal;
    } else {
      // space().message() << "Free IFPSolver other internal";
      m_multi_impl->release<BackEnd::tag::ifpsolver>();
    }
  }

  // Check non used internal
  ALIEN_ASSERT((!MatrixInternal::isInstancied()),
      ("Inconsistent error : IFPSolver Matrix internal is still alive"));

  // New internal
  m_internal = new MatrixInternal(m_multi_impl);

  m_node_list_ref = nodeList;

  /*
   * compute the size of matrix row :
   * for one link (ij), there is one non zero entry on row(i) and one on row(j)
   * be careful store row size on (node+1)
   * row_size is allocated as fortran array, starting at index 1
   */
  Integer* row_size = new Integer[local_nodes_num + 1];

  for (Integer node = 0; node <= local_nodes_num; ++node) {
    row_size[node] = 0;
  }

  typedef std::set<Integer> SetType;
  typedef SetType::iterator SetIterType;
  typedef std::pair<SetIterType, bool> InsertReturnType;

  UniqueArray<SetType> m_matrix(local_nodes_num);

  for (Integer link = 0; link < links_num; ++link) {
    const Integer rowi = nodeToLocalRow[i_node[link]];

    if (rowi != -1) {
      SetType& s = m_matrix[rowi];
      Integer colj = j_node[link];
      InsertReturnType valuej = s.insert(colj);
      if (valuej.second) {
        row_size[rowi + 1]++;
      }

      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        Integer col = ass_elem_node[k];
        InsertReturnType value = s.insert(col);
        if (value.second) {
          row_size[rowi + 1]++;
        }
      }
    }

    const Integer rowj = nodeToLocalRow[j_node[link]];

    if (rowj != -1) {
      std::set<Integer>& s = m_matrix[rowj];
      Integer coli = i_node[link];
      InsertReturnType valuei = s.insert(coli);
      if (valuei.second) {
        row_size[rowj + 1]++;
      }
      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        Integer col = ass_elem_node[k];
        InsertReturnType value = s.insert(col);
        if (value.second) {
          row_size[rowj + 1]++;
        }
      }
    }
  }

  // Compte Column Index Pointer for CSR Format
  Integer* columnIndexPtr = new Integer[local_nodes_num + 1];

  columnIndexPtr[0] = 0;
  m_matrix[0].clear();

  for (Integer node = 1; node < local_nodes_num; ++node) {
    columnIndexPtr[node] = columnIndexPtr[node - 1] + row_size[node];
    row_size[node] = 0;
    m_matrix[node].clear();
  }
  columnIndexPtr[local_nodes_num] =
      columnIndexPtr[local_nodes_num - 1] + row_size[local_nodes_num];

  // Fill ColumnIndex Array
  Integer columnIndexSize = columnIndexPtr[local_nodes_num];

  Integer* columnIndex = new Integer[columnIndexSize];

  // Loop in Link (ij) : on row i, ColIndew=j and on row j ColIndex=i
  for (Integer link = 0; link < links_num; ++link) {
    const Integer rowi = nodeToLocalRow[i_node[link]];

    if (rowi != -1) {
      std::set<Integer>& s = m_matrix[rowi];
      const Integer colj = j_node[link];

      if (s.find(colj) == s.end()) {
        s.insert(colj);
        columnIndex[columnIndexPtr[rowi] + row_size[rowi]] = global_j_node[link];
        row_size[rowi]++;
      }

      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        const Integer col = ass_elem_node[k];
        if (s.find(col) == s.end()) {
          s.insert(col);
          columnIndex[columnIndexPtr[rowi] + row_size[rowi]] = global_ass_elem_node[k];
          row_size[rowi]++;
        }
      }
    }

    const Integer rowj = nodeToLocalRow[j_node[link]];

    if (rowj != -1) {
      std::set<Integer>& s = m_matrix[rowj];
      const Integer coli = i_node[link];

      if (s.find(coli) == s.end()) {
        s.insert(coli);
        columnIndex[columnIndexPtr[rowj] + row_size[rowj]] = global_i_node[link];
        row_size[rowj]++;
      }

      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        const Integer col = ass_elem_node[k];
        if (s.find(col) == s.end()) {
          s.insert(col);
          columnIndex[columnIndexPtr[rowj] + row_size[rowj]] = global_ass_elem_node[k];
          row_size[rowj]++;
        }
      }
    }
  }

  //! extra equations management
  if (global_extra_eq_num > 0) {

    Integer perf_node_num = extra_eq_elem_node_ptr[extra_eq_num];

    Integer* perf_node = new Integer[perf_node_num];
    for (Integer perf = 0; perf < perf_node_num; ++perf)
      perf_node[perf] = extra_eq_elem_node[perf];

    // IFP Solver init
    F2C(ifpsolverinitgraph4)
    (&global_nodes_num, &local_nodes_num, &nodes_num, &max_node_id, &links_num,
        &ass_elem_node_ptr[links_num], &columnIndexSize, columnIndex, columnIndexPtr,
        nodeList, rowUidList, i_node, j_node, global_i_node, global_j_node,
        ass_elem_node_ptr, ass_elem_node, global_ass_elem_node, &global_extra_eq_num,
        &extra_eq_num, &perf_node_num, extra_eq_ids, perf_node, extra_eq_elem_node_ptr,
        &equations_num, &unknowns_num);
    delete[] perf_node;
  } else {
    // IFP Solver init
    F2C(ifpsolverinitgraph2)
    (&global_nodes_num, &local_nodes_num, &nodes_num, &max_node_id, &links_num,
        &ass_elem_node_ptr[links_num], &columnIndexSize, columnIndex, columnIndexPtr,
        nodeList, rowUidList, i_node, j_node, global_i_node, global_j_node,
        ass_elem_node_ptr, ass_elem_node, global_ass_elem_node, &equations_num,
        &unknowns_num);
  }
  delete[] row_size;
  delete[] columnIndex;
  delete[] columnIndexPtr;

  m_graph_initialized = true;

  return true;
}

// La version pour la Taille Block Variable
/*---------------------------------------------------------------------------*/

bool
IFPMatrix::initMatrix(Integer links_num, // num of links
    Integer global_nodes_num, // global num of nodes
    Integer nodes_num, // num of nodes [includes ghost nodes]
    Integer local_nodes_num, // num of nodes in linear system [without ghost nodes]
    Integer max_node_id,
    Integer equations_num, // num of equations by link
    Integer unknowns_num, // num of unknowns by link
    Integer* unknowns_num_per_cell, // num of unknowns per cell
    Integer* nodeList, // list of nodes as local Arcane num [size=nodes_num]
    Integer* globalNodeList, // list of nodes as global Arcane num [size=nodes_num]
    Integer* nodeToLocalRow,
    Integer* i_node, // list of nodes i of links (ij) [local index; size=links_num]
    Integer* global_i_node, //    idem but global index
    Integer* j_node, // list of nodes i of links (ij)
    Integer* global_j_node, // idem but global index
    Integer* ass_elem_node_ptr, // List of offset of Associated element List
    Integer* ass_elem_node, // List of Associated elements of links
    Integer* global_ass_elem_node, // idem with global index
    Integer extra_eq_num, Integer global_extra_eq_num, Integer* extra_eq_ids,
    Integer* extra_eq_elem_node_ptr, Integer* extra_eq_elem_node,
    Integer* extra_eq_elem_lid)
{

  if (MatrixInternal::isInstancied()) {
    if (MatrixInternal::multiImpl() == m_multi_impl) {
      // space().message() << "Free IFPSolver owned internal";
      delete m_internal;
    } else {
      // space().message() << "Free IFPSolver other internal";
      m_multi_impl->release<BackEnd::tag::ifpsolver>();
    }
  }

  ALIEN_ASSERT((!MatrixInternal::isInstancied()),
      ("Inconsistent error : IFPSolver Matrix internal is still alive"));

  m_internal = new MatrixInternal(m_multi_impl);

  m_node_list_ref = nodeList;

  Integer* row_size = new Integer[local_nodes_num + 1];

  for (Integer node = 0; node <= local_nodes_num; ++node) {
    row_size[node] = 0;
  }

  UniqueArray<std::set<Integer>> m_matrix(local_nodes_num);

  for (Integer link = 0; link < links_num; ++link) {
    const Integer rowi = nodeToLocalRow[i_node[link]];

    if (rowi != -1) {
      std::set<Integer>& s = m_matrix[rowi];
      Integer colj = j_node[link];
      if (s.find(colj) == s.end()) {
        s.insert(colj);
        row_size[rowi + 1]++;
      }

      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        Integer col = ass_elem_node[k];
        if (s.find(col) == s.end()) {
          s.insert(col);
          row_size[rowi + 1]++;
        }
      }
    }

    const Integer rowj = nodeToLocalRow[j_node[link]];

    if (rowj != -1) {
      std::set<Integer>& s = m_matrix[rowj];
      Integer coli = i_node[link];
      if (s.find(coli) == s.end()) {
        s.insert(coli);
        row_size[rowj + 1]++;
      }
      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        Integer col = ass_elem_node[k];
        if (s.find(col) == s.end()) {
          s.insert(col);
          row_size[rowj + 1]++;
        }
      }
    }
  }

  Integer* columnIndexPtr = new Integer[local_nodes_num + 1];

  columnIndexPtr[0] = 0;
  m_matrix[0].clear();

  for (Integer node = 1; node < local_nodes_num; ++node) {
    columnIndexPtr[node] = columnIndexPtr[node - 1] + row_size[node];
    row_size[node] = 0;
    m_matrix[node].clear();
  }
  columnIndexPtr[local_nodes_num] =
      columnIndexPtr[local_nodes_num - 1] + row_size[local_nodes_num];

  Integer columnIndexSize = columnIndexPtr[local_nodes_num];

  Integer* columnIndex = new Integer[columnIndexSize];

  for (Integer link = 0; link < links_num; ++link) {
    const Integer rowi = nodeToLocalRow[i_node[link]];

    if (rowi != -1) {
      std::set<Integer>& s = m_matrix[rowi];
      const Integer colj = j_node[link];

      if (s.find(colj) == s.end()) {
        s.insert(colj);
        columnIndex[columnIndexPtr[rowi] + row_size[rowi]] = global_j_node[link];
        row_size[rowi]++;
      }

      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        const Integer col = ass_elem_node[k];
        if (s.find(col) == s.end()) {
          s.insert(col);
          columnIndex[columnIndexPtr[rowi] + row_size[rowi]] = global_ass_elem_node[k];
          row_size[rowi]++;
        }
      }
    }

    const Integer rowj = nodeToLocalRow[j_node[link]];

    if (rowj != -1) {
      std::set<Integer>& s = m_matrix[rowj];
      const Integer coli = i_node[link];

      if (s.find(coli) == s.end()) {
        s.insert(coli);
        columnIndex[columnIndexPtr[rowj] + row_size[rowj]] = global_i_node[link];
        row_size[rowj]++;
      }

      for (Integer k = ass_elem_node_ptr[link]; k < ass_elem_node_ptr[link + 1]; ++k) {
        const Integer col = ass_elem_node[k];
        if (s.find(col) == s.end()) {
          s.insert(col);
          columnIndex[columnIndexPtr[rowj] + row_size[rowj]] = global_ass_elem_node[k];
          row_size[rowj]++;
        }
      }
    }
  }

  // les puits couplés
  if (global_extra_eq_num > 0) {
    Integer perf_node_num = extra_eq_elem_node_ptr[extra_eq_num];

    Integer* perf_node = new Integer[perf_node_num];
    Integer* perf_lid = new Integer[perf_node_num];
    for (Integer perf = 0; perf < perf_node_num; ++perf) {
      perf_node[perf] = extra_eq_elem_node[perf];
      perf_lid[perf] = extra_eq_elem_lid[perf];
    }

    F2C(ifpsolverinitgraph6)
    (&global_nodes_num, &local_nodes_num, &nodes_num, &max_node_id, &links_num,
        &ass_elem_node_ptr[links_num], &columnIndexSize, columnIndex, columnIndexPtr,
        nodeList, globalNodeList, i_node, j_node, global_i_node, global_j_node,
        ass_elem_node_ptr, ass_elem_node, global_ass_elem_node, &global_extra_eq_num,
        &extra_eq_num, &perf_node_num, extra_eq_ids, perf_node, perf_lid,
        extra_eq_elem_node_ptr, &equations_num, &unknowns_num, unknowns_num_per_cell);
    delete[] perf_node;
    delete[] perf_lid;
  } else {
    // sans les puits couplés
    F2C(ifpsolverinitgraph7)
    (&global_nodes_num, &local_nodes_num, &nodes_num, &max_node_id, &links_num,
        &ass_elem_node_ptr[links_num], &columnIndexSize, columnIndex, columnIndexPtr,
        nodeList, globalNodeList, i_node, j_node, global_i_node, global_j_node,
        ass_elem_node_ptr, ass_elem_node, global_ass_elem_node, &equations_num,
        &unknowns_num, unknowns_num_per_cell);
  }
  delete[] row_size;
  delete[] columnIndex;
  delete[] columnIndexPtr;

  m_graph_initialized = true;

  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::initMatrix(int equations_num, int unknowns_num, int global_nodes_num,
    int nodes_num, int row_offset, ConstArrayView<Integer> columnIndexesPtr,
    ConstArrayView<Integer> columnIndexes, Int64 timestamp)
{
  /* m_node_to_local_row describes local nodes only.
   * Ghost nodes verifies m_node_to_local_row[node] == -1
   * Then, on the local matrix, for each link, only well defined
   * nodes (m_node_to_local_row != -1) is used as row index.
   * If a link(i,j) has a ghost node j, it will be duplicated as link(i,j)
   * on owner(j) with with i as a ghost node.
   */

  if (MatrixInternal::isInstancied()) {
    if (MatrixInternal::multiImpl() == m_multi_impl) {
      // space().message() << "Free IFPSolver owned internal";
      // delete m_internal;
      if (timestamp == m_internal->timestamp()) {
        alien_debug([&] {
          cout() << " KEEP IFP INTERNAL" << timestamp << " " << m_internal->timestamp();
        });

        m_internal->init();
        return true;
      } else
        delete m_internal;
    } else {
      // space().message() << "Free IFPSolver other internal";
      m_multi_impl->release<BackEnd::tag::ifpsolver>();
    }
  }

  // Check non used internal
  ALIEN_ASSERT((!MatrixInternal::isInstancied()),
      ("Inconsistent error : IFPSolver Matrix internal is still alive"));

  // New internal
  m_internal = new MatrixInternal(m_multi_impl, timestamp);

  int matrix_size = columnIndexesPtr[nodes_num];

  UniqueArray<Integer> split_tags;
  m_internal->m_elliptic_split_tag = computeEllipticSplitTags(split_tags, equations_num);

  //     =====    IFP Solver init   ==================================
  // Dans le cas de CprAMG scalaire, on utilise
  // un tableau de tags (m_split_tags) permettant de separer le bloc elliptique
  // du reste.
  if (m_internal->m_elliptic_split_tag) {
    F2C(ifpsolverinitgraph3)
    (&global_nodes_num, &nodes_num, &nodes_num, &matrix_size, &row_offset,
        columnIndexesPtr.data(), columnIndexes.data(), &equations_num, &unknowns_num,
        split_tags.data());
  } else {
    F2C(ifpsolverinitgraph3)
    (&global_nodes_num, &nodes_num, &nodes_num, &matrix_size, &row_offset,
        columnIndexesPtr.data(), columnIndexes.data(), &equations_num, &unknowns_num);
  }
  m_graph_initialized = true;
  return true;
}

/*
void
IFPMatrix::
initWellMatrix(
     Integer extra_eq_num,
     Integer global_extra_eq_num,
     Integer* extra_eq_ids,
     Integer* extra_eq_elem_node_ptr,
     Integer* extra_eq_elem_node,
     Integer* extra_eq_elem_lid
     )
{
   //les puits couplés
    if(global_extra_eq_num>0)
    {
      Integer perf_node_num = extra_eq_elem_node_ptr[extra_eq_num] ;

      std::vector<Integer> perf_node(perf_node_num);
      std::vector<Integer> perf_lid(perf_node_num);
      for(Integer perf=0;perf<perf_node_num;++perf)
      {
        perf_node[perf] = extra_eq_elem_node[perf] ;
        perf_lid[perf] = extra_eq_elem_lid[perf] ;
      }

      F2C(ifpsolverinitwellgraph)(
                               &global_extra_eq_num,
                               &extra_eq_num,
                               &perf_node_num,
                               extra_eq_ids,
                               perf_node.data(),
                               perf_lid.data(),
                               extra_eq_elem_node_ptr);
    }
}
*/

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::allocate()
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsolverallocatematrix)();
  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::allocate(ArrayView<Integer> coupled_lids)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  int nwells = coupled_lids.size();
  UniqueArray<Integer> well_lids(nwells);
  for (Integer i = 0; i < nwells; ++i) {
    well_lids[i] = coupled_lids[i] + 1;
  }
  F2C(ifpsolverinitcoupledwells)(&nwells, well_lids.data());

  F2C(ifpsolverallocatematrix)();
  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::allocateRS(ArrayView<Integer> coupled_uids)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  int nwells = coupled_uids.size();
  UniqueArray<Integer> well_uids(nwells);
  for (Integer i = 0; i < nwells; ++i)
    well_uids[i] = coupled_uids[i] + 1;
  F2C(ifpsolverinitrscoupledwells)(&nwells, well_uids.data());

  F2C(ifpsolverallocatematrixrs)();
  return true;
}

/*---------------------------------------------------------------------------*/

void
IFPMatrix::freeData()
{
  F2C(ifpmatrixfreedata)();
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::setMatrixValues(const double* values)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsolversetmatrixdata)((double*)values);
  m_internal->m_filled = true;

  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::setMatrixBlockValues(const double* values)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsolversetmatrixdata)((double*)values);
  m_internal->m_filled = true;

  return true;
}

/*---------------------------------------------------------------------------*/
// A. Anciaux pour BlockTailleVariable
bool
IFPMatrix::setMatrixRsValues(Real* dFijdXi, Real* dCidXi, Integer* nodeList,
    Integer* nodeToLocalRow, Integer* i_node, Integer* j_node)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsetmatrixrsdata)(dFijdXi, dCidXi, nodeList, i_node, j_node, nodeToLocalRow);
  m_internal->m_filled = true;
  m_internal->m_system_is_resizeable = true;

  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::setMatrixValues(Real* dFijdXi, Real* dCidXi, Integer* nodeList,
    Integer* nodeToLocalRow, Integer* i_node, Integer* j_node)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsetmatrixdata)(dFijdXi, dCidXi, nodeList, i_node, j_node, nodeToLocalRow);
  m_internal->m_filled = true;

  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::setMatrixValues(Real* dFijdXi, Real* dCidXi, Integer* nodeList,
    Integer* nodeToLocalRow, Integer* i_node, Integer* j_node, Integer* ass_elem_node_ptr)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsolversetsumfirsteq)(&m_sum_first_eq);
  F2C(ifpsetmatrixdata2)
  (dFijdXi, dCidXi, nodeList, i_node, j_node, ass_elem_node_ptr, nodeToLocalRow);
  m_internal->m_filled = true;

  return true;
}

bool
IFPMatrix::setMatrixValues(Real* dFijdXi1, Real* dFijdXi2, Real* dCidXi,
    Integer* nodeList, Integer* nodeToLocalRow, Integer* i_node, Integer* j_node,
    Integer* ass_elem_node_ptr, Integer NumOfLink1, Integer NumOfLink2)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsolversetsumfirsteq)(&m_sum_first_eq);
  F2C(ifpsetmatrixdata4)
  (dFijdXi1, dFijdXi2, dCidXi, nodeList, i_node, j_node, ass_elem_node_ptr,
      nodeToLocalRow, &NumOfLink1, &NumOfLink2);
  m_internal->m_filled = true;

  return true;
}
/*---------------------------------------------------------------------------*/

bool
IFPMatrix::setMatrixExtraValues(
    Real* ExtraRowValues, Real* ExtraColValues, Real* ExtraDiagValues)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  // space().message() << "IFP MATRIX SET WELL DATA";
  F2C(ifpsetmatrixwelldata)(ExtraRowValues, ExtraColValues, ExtraDiagValues);
  m_internal->m_extra_filled = true;
  // space().message() << "IFP MATRIX AFTER SET WELL DATA";

  return true;
}

/*---------------------------------------------------------------------------*/
// A. Anciaux pour BlockTailleVariable
bool
IFPMatrix::setMatrixRsExtraValues(
    Real* ExtraRowValues, Real* ExtraColValues, Real* ExtraDiagValues)
{
  if (!MatrixInternal::isInstancied())
    throw FatalErrorException(
        A_FUNCINFO, "Cannot set values in non-allocated IFPSolver matrix");

  F2C(ifpsetmatrixrswelldata)(ExtraRowValues, ExtraColValues, ExtraDiagValues);
  m_internal->m_extra_filled = true;

  return true;
}

/*
bool
IFPMatrix::
initSubMatrix01( int nrow,
                 int global_nb_extra_eq,
                 int nb_extra_eq,
                 int const* row_offset,
                 int const* cols)
{
  int* extra_eq_ids = nullptr;
  int* cols_lid = nullptr;
  initWellMatrix(nb_extra_eq,
                 global_nb_extra_eq,
                 extra_eq_ids,
                 (int*) row_offset,
                 (int*) cols,
                 cols_lid) ;
  return true ;
}

bool
IFPMatrix::
initSubMatrix11(int nb_extra_eq,
                int const* row_offset,
                int const* cols)
{
  return true ;
}


bool
IFPMatrix::
initSubMatrix11Values( Real const* ExtraDiagValues)
{
  F2C(ifpsetmatrixwelldiagdata)((Real*)ExtraDiagValues);
  return true ;
}

bool
IFPMatrix::
initSubMatrix10Values( Real const* ExtraRowValues)
{
  F2C(ifpsetmatrixwellrowdata)((Real*)ExtraRowValues);
  return true ;
}

bool
IFPMatrix::
initSubMatrix01Values( Real const* ExtraColValues)
{
  F2C(ifpsetmatrixwellcoldata)((Real*)ExtraColValues);
  return true ;
}
*/

/*---------------------------------------------------------------------------*/
bool
IFPMatrix::setSymmetricProfile(bool value)
{
  m_symmetric_profile = value;

  return true;
}

/*---------------------------------------------------------------------------*/

bool
IFPMatrix::getSymmetricProfile() const
{
  return m_symmetric_profile;
}

/*---------------------------------------------------------------------------*/
bool
IFPMatrix::computeEllipticSplitTags(
    UniqueArray<Integer>& split_tags, int equation_num) const
{
  // const IIndexManager * iindex_mng = this->space().indexManager();
  const ISpace& space = this->rowSpace();
  const MatrixDistribution& dist = this->distribution();

  // Integer total_size;
  Integer min_local_index = dist.rowOffset();
  Integer local_size = dist.localRowSize();

  // iindex_mng->stats(total_size,min_local_index,local_size);

  bool elliptic_split_tag_found = false;

  split_tags.resize(local_size * equation_num);
  split_tags.fill(1); // DP_NoTyp == 1 cf. M_DOMAINPARTITION_MODULE.f90

  // fill-in m_split_tags
  // if(iindex_mng->enumerateEntry().null()) return elliptic_split_tag_found;
  if (space.nbField() == 0)
    return elliptic_split_tag_found;
  // On tague ici des lignes en passant temporairement par les colonnes qui sont
  // actuellement taguées et placées identiquement
  /*
  for(IIndexManager::EntryEnumerator i = iindex_mng->enumerateEntry(); i.hasNext(); ++i)
    {
      UniqueArray<Integer> indices = iindex_mng->getIndexes(*i);
      if(i->tagValue("block-tag") == "Elliptic" and not indices.empty())
        {
          elliptic_split_tag_found = true ;

          ConstArrayView<Integer> localIds = i->getOwnLocalIds();
          for(Integer i=0,is=localIds.size();i<is;i++){
            const Integer index=indices[localIds[i]];
            // par construction les OwnLocalIds sont toujours isOwn==true
            split_tags[(index-min_local_index)*equation_num] = 2 ; // DP_PressTyp ==  cf.
  M_DOMAINPARTITION_MODULE.f90
          }
        }
    }
  */
  for (Integer i = 0; i < space.nbField(); ++i) {
    const UniqueArray<Integer>& indices = space.field(i);
    if (space.fieldLabel(i) == "Elliptic" and not indices.empty()) {
      elliptic_split_tag_found = true;
      for (Integer j = 0; j < indices.size(); ++j) {
        const Integer index = indices[j];
        split_tags[(index - min_local_index) * equation_num] =
            2; // DP_PressTyp ==  cf. M_DOMAINPARTITION_MODULE.f90
      }
    }
  }

  return elliptic_split_tag_found;
}

/*---------------------------------------------------------------------------*/

void
IFPMatrix::freeGraphData()
{
  if (!m_graph_initialized)
    return;

  freeData();

  delete m_internal;

  m_graph_initialized = false;
}

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
