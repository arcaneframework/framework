// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
void case1c() 
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
    // C: Try to use 2 parallelMng for filling-in.

    IParallelMng* areaU_pm = m_parallel_mng->
      createSubParallelMng(owners(areaU));
    IParallelMng* areaU_pm = m_parallel_mng->
      createSubParallelMng(owners(areaP));
    
    MatrixEditor edit_aU(m_a, index_manager, areaU_pm);
    MatrixEditor edit_aP(m_a, index_manager, areaP_pm);
    // edit_a is a "copy (CoW ?)" of m_a data.
    // m_a contents is thus always valid !
    // C: areaU_pm is the parallel mng where edit_a is defined.

    if (areaU_pm->commRank() >= 0) { // areaU is defined there.
      ENUMERATE_CELL(icell, areaU.own()) {
	edit_aU(icell,icell) = fij(icell, icell);
	ENUMERATE_SUB_ITEM(Cell, isubcell, icell) {
	  edit_aU(icell, isubcell) += fij(icell, isubcell);
	}
      }
    }
    if (areaP_pm->commRank() >= 0) { // areaP is defined there.
      ENUMERATE_NODE(inode, areaP.own()) {
	ENUMERATE_SUB_ITEM(Cell, isubcell, inode) {
	  edit_aP(inode, isubcell) += fij(inode, isubcell);
	}
      }
    }

    // C: How to avoid inter-locking?
    m_a.add_contrib(edit_aU);
    m_a.add_contrib(edit_aP);
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
