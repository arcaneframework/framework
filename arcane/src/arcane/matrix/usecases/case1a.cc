// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
void case1a() 
{
  IAlgebraicMng *alg_mng = createAlgebraicMng(m_parallel_mng);

  // Not the perfect way to create space
  IndexedSpace space = alg_mng->createSpace("Nom", index_manager->global_size());

  Vector v_b(space);
  Vector v_x(space);
  Matrix m_a(space,space);

  // Fill matrix m_a
  {
    MatrixIJBuilder build_a(m_a, index_manager);

    // Cell-Cell contribution on area U.
    ENUMERATE_CELL(icell, areaU.own()) {
      build_a(icell,icell) += fij(icell, icell);
      ENUMERATE_SUB_ITEM(Cell, isubcell, icell) {
	build_a(icell, isubcell) += fij(icell, isubcell);
      }
    }
    
    ENUMERATE_NODE(inode, areaP.own()) {
      ENUMERATE_SUB_ITEM(Cell, isubcell, inode) {
	build_a(inode, isubcell) += fij(inode, isubcell);
      }
    }
    //End of scope for build_a -> matrix build.
    // Not a good idea ...
  }
  
  // Fill vector v_b
  {
    VectorAccessor va_b (v_b, index_manager);

    ENUMERATE_CELL(icell, areaU.own()) {
      va_b(icell) = var_cell[icell];
    }
    ENUMERATE_NODE(inode, areaP.own()) {
      va_b(inode) = var_node[inode];
    }
    //End of scope for va_b -> vector build.
    // Not a good idea ...
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
