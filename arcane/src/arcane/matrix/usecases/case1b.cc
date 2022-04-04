// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
void case1b() 
{
  IAlgebraicMng *alg_mng = createAlgebraicMng(m_parallel_mng);

  // Not the perfect way to create space
  IndexedSpace space = alg_mng->createSpace("Nom",
					    index_manager->global_size());

  Vector v_b(space);
  Vector v_x(space);
  Matrix m_a(space,space);

  // Fill matrix m_a
  {
    MatrixEditor edit_a(m_a, index_manager);
    // edit_a is a "copy (CoW ?)" of m_a data.
    // m_a contents is thus always valid !

    // Cell-Cell contribution on area U.
    ENUMERATE_CELL(icell, areaU.own()) {
      edit_a(icell,icell) = fij(icell, icell);
      ENUMERATE_SUB_ITEM(Cell, isubcell, icell) {
	edit_a(icell, isubcell) += fij(icell, isubcell);
      }
    }
    
    ENUMERATE_NODE(inode, areaP.own()) {
      ENUMERATE_SUB_ITEM(Cell, isubcell, inode) {
	edit_a(inode, isubcell) += fij(inode, isubcell);
      }
    }

    m_a.replace(edit_a);
    // m_a.add_contrib(build_a);
  }
  
  // Fill vector v_b
  {
    VectorEditor ve_b (v_b, index_manager);

    ENUMERATE_CELL(icell, areaU.own()) {
      ve_b(icell) = var_cell[icell];
    }
    ENUMERATE_NODE(inode, areaP.own()) {
      ve_b(inode) = var_node[inode];
    }
    v_b.replace(ve_b);
  }
  
  Solver solver = solverMng->createSolver(alg_mng, options);
  v_x = solver.solve(m_a, v_b);

  if (!solver.failed()) {
    Fatal("Unable to solve Ax=b");
  }
  
  // res is automatically in the good space.
  Vector v_residual = v_b - m_a*v_x;
  
  info() << "Residual norm_2 = " << norm2(v_residual);
}
