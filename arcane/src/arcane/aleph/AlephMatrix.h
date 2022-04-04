﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephMatrix.h                                                    (C) 2010 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_MATRIX_H
#define ALEPH_MATRIX_H

#include <map>
#include "arcane/aleph/AlephGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

class IAlephMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Matrice d'un système linéaire.
 */
class ARCANE_ALEPH_EXPORT AlephMatrix
: public TraceAccessor
{
 public:
  AlephMatrix(AlephKernel*);
  ~AlephMatrix();

 public:
  void create(void);
  void create(IntegerConstArrayView, bool = false);
  void create_really(void);
  void reset(void);
  Integer reIdx(Integer, Array<Int32*>&);
  void reSetValuesIn(AlephMatrix*, Array<Int32*>&);
  void reAddValuesIn(AlephMatrix*, Array<Int32*>&);
  void updateKnownRowCol(Integer, Integer, Real);
  void rowMapMapCol(Integer, Integer, Real);
  void addValue(const VariableRef&, const Item&,
                const VariableRef&, const Item&, const Real);
  void addValue(const VariableRef&, const ItemEnumerator&,
                const VariableRef&, const ItemEnumerator&, const Real);
  void setValue(const VariableRef&, const Item&,
                const VariableRef&, const Item&, const Real);
  void setValue(const VariableRef&, const ItemEnumerator&,
                const VariableRef&, const ItemEnumerator&, const Real);
  void addValue(Integer, Integer, Real);
  void setValue(Integer, Integer, Real);
  void writeToFile(const String);
  void startFilling();
  void assemble();
  void assemble_waitAndFill();
  void reassemble(Integer&, Real*);
  void reassemble_waitAndFill(Integer&, Real*);
  void solve(AlephVector*, AlephVector*, Integer&, Real*, AlephParams*, bool = false);
  void solveNow(AlephVector*, AlephVector*, AlephVector*, Integer&, Real*, AlephParams*);

 private:
  AlephKernel* m_kernel;
  Integer m_index;
  ArrayView<Integer> m_ranks;
  bool m_participating_in_solver;
  IAlephMatrix* m_implementation;

 private:
  // Matrice utilisée dans le cas où nous sommes le solveur
  MultiArray2<Int32> m_aleph_matrix_buffer_rows;
  MultiArray2<Int32> m_aleph_matrix_buffer_cols;
  MultiArray2<Real> m_aleph_matrix_buffer_vals;
  // Tableaux tampons des setValues
  Integer m_setValue_idx;
  UniqueArray<Int32> m_setValue_row;
  UniqueArray<Int32> m_setValue_col;
  UniqueArray<Real> m_setValue_val;

 private: // Tableaux tampons des addValues
  typedef std::map<Integer, Integer> colMap;
  typedef std::map<Integer, colMap*> rowColMap;
  rowColMap m_row_col_map;
  Integer m_addValue_idx;
  UniqueArray<Integer> m_addValue_row;
  UniqueArray<Integer> m_addValue_col;
  UniqueArray<Real> m_addValue_val;

 private: // Tableaux des requètes
  UniqueArray<Parallel::Request> m_aleph_matrix_mpi_data_requests;
  UniqueArray<Parallel::Request> m_aleph_matrix_mpi_results_requests;

 private: // Résultats. Placés ici afin de les conserver hors du scope de la fonction les utilisant
  UniqueArray<Int32> m_aleph_matrix_buffer_n_iteration;
  UniqueArray<Real> m_aleph_matrix_buffer_residual_norm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
