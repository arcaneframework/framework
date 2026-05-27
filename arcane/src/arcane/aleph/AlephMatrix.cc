// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephMatrix.cc                                              (C) 2000-2024 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephArcane.h"
#include "arcane/MeshVariableScalarRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/******************************************************************************
 *****************************************************************************/
AlephMatrix::
AlephMatrix(AlephKernel* kernel)
: TraceAccessor(kernel->parallel()->traceMng())
, m_kernel(kernel)
, m_index(kernel->index())
, m_setValue_idx(0)
, m_addValue_idx(0)
{
  ItacFunction(AlephMatrix);
  if (kernel->isInitialized() == false) {
    debug() << "\33[1;32m[AlephMatrix::AlephMatrix] New Aleph matrix, but kernel is not initialized!\33[0m";
    return;
  }
  // Retrieving the ranks used for this resolution
  m_ranks = kernel->solverRanks(m_index);
  // Boolean to know if we participate or not
  m_participating_in_solver = kernel->subParallelMng(m_index) != NULL;
  debug() << "\33[1;32m[AlephMatrix::AlephMatrix] New Aleph matrix\33[0m";
  if (!m_participating_in_solver) {
    debug() << "\33[1;32m[AlephMatrix::AlephMatrix] Not concerned by this one!\33[0m";
    return;
  }
  debug() << "\33[1;32m[AlephMatrix::AlephMatrix] site size="
          << m_kernel->subParallelMng(m_index)->commSize()
          << " @"
          << m_kernel->subParallelMng(m_index)->commRank() << "\33[0m";
  // We are going to look for a matrix from the factory which provides the interface to external libraries
  m_implementation = m_kernel->factory()->GetMatrix(m_kernel, m_index);
  traceMng()->flush();
}

/******************************************************************************
 *****************************************************************************/
AlephMatrix::
~AlephMatrix()
{
  ItacFunction(AlephMatrix);
  debug() << "\33[1;32m\t\t[~AlephMatrix]\33[0m";
  rowColMap::const_iterator i = m_row_col_map.begin();
  for (; i != m_row_col_map.end(); ++i)
    delete i->second;
}

/******************************************************************************
 * Matrix 'create' with the 'void' API
 * BaseForm[Hash["AlephMatrix::create(void)", "CRC32"], 16] = fff06e2
 *****************************************************************************/
void AlephMatrix::create(void)
{
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::create");
  debug() << "\33[1;32m[AlephMatrix::create(void)]\33[0m";
  // If the kernel is not initialized, we have nothing to do
  if (!m_kernel->isInitialized())
    return;
  // If there are 'others' and we are not part of them,
  // we broadcast that a 'create' needs to be done
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther())
    m_kernel->world()->broadcast(UniqueArray<unsigned long>(1, 0xfff06e2l).view(), 0);
  // We flush in anticipation of filling, it must be done even if configured
  // However, we do not flush the one for addValue
  m_setValue_idx = 0;
}

/******************************************************************************
 * Matrix 'create' with the API that specifies the number of non-zero elements per row
 * BaseForm[Hash["AlephMatrix::create(IntegerConstArrayView,bool)", "CRC32"], 16] = 5c3111b1
 *****************************************************************************/
void AlephMatrix::
create(IntegerConstArrayView row_nb_element, bool has_many_elements)
{
  ARCANE_UNUSED(row_nb_element);
  ARCANE_UNUSED(has_many_elements);
  debug() << "\33[1;32m[AlephMatrix::create(old API)] API with row_nb_element + has_many_elements\33[0m";
  this->create();
}

/*!
 * \brief reset to flush the [set&add]Value arrays
 */
void AlephMatrix::
reset(void)
{
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::reset");
  debug() << "\33[1;32m[AlephMatrix::reset]\33[0m";
  m_setValue_val.fill(0.0);
  m_addValue_val.fill(0.0);
}

/*!
 * \brief addValue from arguments in IVariables, Items, and Real
 *****************************************************************************/
void AlephMatrix::
addValue(const VariableRef& rowVar, const ItemEnumerator& rowItm,
         const VariableRef& colVar, const ItemEnumerator& colItm,
         const Real val)
{
  addValue(rowVar, *rowItm, colVar, *colItm, val);
}
void AlephMatrix::addValue(const VariableRef& rowVar, const Item& rowItm,
                           const VariableRef& colVar, const Item& colItm,
                           const Real val)
{
  AlephInt row = m_kernel->indexing()->get(rowVar, rowItm);
  AlephInt col = m_kernel->indexing()->get(colVar, colItm);
  if (m_kernel->isInitialized()) {
    const AlephInt row_offset = m_kernel->topology()->part()[m_kernel->rank()];
    row += row_offset;
    col += row_offset;
  }
  //debug()<<"[AlephMatrix::addValue] IVariable/Item add @ ["<<row<<","<<col<<"]="<<val;
  addValue(row, col, val);
}

void AlephMatrix::
updateKnownRowCol(Integer row,
                  Integer col,
                  Real val)
{
  //debug()<<"\33[1;32m[AlephMatrix::updateKnownRowCol]\33[0m";
  m_addValue_row.add(row);
  m_addValue_col.add(col);
  m_addValue_val.add(val);
  m_addValue_idx += 1;
  // We do the same on the 'set' side to have the correct size
  m_setValue_row.add(row);
  m_setValue_col.add(col);
  m_setValue_val.add(0.);
}

void AlephMatrix::
rowMapMapCol(Integer row,
             Integer col,
             Real val)
{
  rowColMap::const_iterator iRowMap = m_row_col_map.find(row);
  // If the row is not even known yet
  // We add a map entry (map(m_addValue_idx))
  if (iRowMap == m_row_col_map.end()) {
    colMap* jMap = new colMap();
    /*debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] row "
          <<row<<" inconue, m_addValue_idx="
          <<m_addValue_idx<<"\33[0m";*/
    m_row_col_map.insert(std::make_pair(row, jMap));
    jMap->insert(std::make_pair(col, m_addValue_idx));
    updateKnownRowCol(row, col, val);
    return;
  }
  // We focus on the second dimension
  colMap* jMap = iRowMap->second;
  colMap::const_iterator iColMap = jMap->find(col);
  // If this column is not known for this row
  // We add an entry
  if (iColMap == jMap->end()) {
    /*debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] col "
          <<col<<" inconue, m_addValue_idx="
          <<m_addValue_idx<<"\33[0m";*/
    jMap->insert(std::make_pair(col, m_addValue_idx));
    updateKnownRowCol(row, col, val);
    return;
  }
  // Otherwise we add
  //debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] hit\33[0m";
  //debug()<<"[AlephMatrix::rowMapMapCol] += for ["<<row<<","<<col<<"]="<<val; traceMng()->flush();
  m_addValue_val[iColMap->second] += val;
}

/*!
 * \brief standard addValue in (i,j,val)
 */
void AlephMatrix::
addValue(Integer row, Integer col, Real val)
{
  //debug()<<"\33[32m[AlephMatrix::addValue] addValue("<<row<<","<<col<<")="<<val<<"\33[0m";
  row = m_kernel->ordering()->swap(row);
  col = m_kernel->ordering()->swap(col);
  // Search for the cell (row,j) if it already exists
  rowMapMapCol(row, col, val);
}

/*!
 * \brief setValue from arguments in IVariables, ItemEnumerator, and Real
 */
void AlephMatrix::
setValue(const VariableRef& rowVar, const ItemEnumerator& rowItm,
         const VariableRef& colVar, const ItemEnumerator& colItm,
         const Real val)
{
  setValue(rowVar, *rowItm, colVar, *colItm, val);
}

/*!
 * \brief setValue from arguments in IVariables, Items, and Real
 */
void AlephMatrix::
setValue(const VariableRef& rowVar, const Item& rowItm,
         const VariableRef& colVar, const Item& colItm,
         const Real val)
{
  Integer row = m_kernel->indexing()->get(rowVar, rowItm);
  Integer col = m_kernel->indexing()->get(colVar, colItm);
  //debug()<<"[AlephMatrix::setValue] dof #"<<m_setValue_idx<<" ["<<row<<","<<col<<"]="<<val;
  if (m_kernel->isInitialized()) {
    const Integer row_offset = m_kernel->topology()->part()[m_kernel->rank()];
    row += row_offset;
    col += row_offset;
  }
  setValue(row, col, val);
}

/*!
 * \brief standard setValue from arguments (row,col,val)
 */
void AlephMatrix::
setValue(Integer row, Integer col, Real val)
{
  // Re-ordering if necessary
  row = m_kernel->ordering()->swap(row);
  col = m_kernel->ordering()->swap(col);
  // If the kernel has already been configured,
  // we ensure that the 'geometry/support' has not changed between resolutions
  if (m_kernel->configured()) {
    if ((m_setValue_row[m_setValue_idx] != row) ||
        (m_setValue_col[m_setValue_idx] != col))
      throw FatalErrorException("Aleph::setValue", "Row|Col have changed!");
    m_setValue_row[m_setValue_idx] = row;
    m_setValue_col[m_setValue_idx] = col;
    m_setValue_val[m_setValue_idx] = val;
  }
  else {
    m_setValue_row.add(row);
    m_setValue_col.add(col);
    m_setValue_val.add(val);
  }
  m_setValue_idx += 1;
}

/*!
 * \brief reIdx searches for the correspondence of the AlephIndexing
 */
Int32 AlephMatrix::
reIdx(Integer ij,
      Array<Int32*>& known_items_own_address)
{
  return *known_items_own_address[ij];
}

/*!
 * \brief reSetValuesIn re-plays the setValue with the indexes calculated via the AlephIndexing
 */
void AlephMatrix::
reSetValuesIn(AlephMatrix* thisMatrix,
              Array<Int32*>& known_items_own_address)
{
  for (Integer k = 0, kMx = m_setValue_idx; k < kMx; k += 1) {
    Integer i = reIdx(m_setValue_row[k], known_items_own_address);
    Integer j = reIdx(m_setValue_col[k], known_items_own_address);
    thisMatrix->setValue(i, j, m_setValue_val[k]);
  }
}

/*!
 * \brief reAddValuesIn re-plays the addValue with the indexes calculated via the AlephIndexing
 */
void AlephMatrix::
reAddValuesIn(AlephMatrix* thisMatrix,
              Array<Int32*>& known_items_own_address)
{
  for (Integer k = 0, kMx = m_addValue_row.size(); k < kMx; k += 1) {
    const Integer row = reIdx(m_addValue_row[k], known_items_own_address);
    const Integer col = reIdx(m_addValue_col[k], known_items_own_address);
    const Real val = m_addValue_val[k];
    thisMatrix->addValue(row, col, val);
  }
}

/*!
 * \brief assemble the matrices before resolution
 */
void AlephMatrix::
assemble(void)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::assemble");
  // If the kernel is not initialized, we still do nothing
  if (!m_kernel->isInitialized()) {
    debug() << "\33[1;32m[AlephMatrix::assemble] Trying to assemble a matrix"
            << "from an uninitialized kernel!\33[0m";
    return;
  }
  // If no [set|add]Value has been received, this is not normal
  if (m_addValue_idx != 0 && m_setValue_idx != 0)
    throw FatalErrorException("AlephMatrix::assemble", "Still exclusives [add||set]Value required!");
  // If addValue have been captured, they must be 're-played'
  // Warning: for now, adds and sets are disjoint!
  if (m_addValue_idx != 0) {
    debug() << "\33[1;32m[AlephMatrix::assemble] m_addValue_idx!=0\33[0m";
    // We flush our setValues index
    m_setValue_idx = 0;
    Timer::Action ta(m_kernel->subDomain(), "Flatenning addValues");
    debug() << "\t\33[32m[AlephMatrix::assemble] Flatenning addValues size=" << m_addValue_row.size() << "\33[0m";
    for (Integer k = 0, kMx = m_addValue_row.size(); k < kMx; ++k) {
      m_setValue_row[k] = m_addValue_row[k];
      m_setValue_col[k] = m_addValue_col[k];
      m_setValue_val[k] = m_addValue_val[k];
      /*debug()<<"\t\33[32m[AlephMatrix::assemble] setValue ("<<m_setValue_row[k]
        <<","<<m_setValue_col[k]<<")="<<m_setValue_val[k]<<"\33[0m";*/
      m_setValue_idx += 1;
    }
  }
  // If there are 'others' and we are not part of them, we inform them of the assembly
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()) {
    debug() << "\33[1;32m[AlephMatrix::assemble] We inform the other kappas that we are assembling"
            << "\33[0m";
    m_kernel->world()->broadcast(UniqueArray<unsigned long>(1, 0x74f253cal).view(), 0);
    // And we give them the info of m_setValue_idx
    m_kernel->world()->broadcast(UniqueArray<Integer>(1, m_setValue_idx).view(), 0);
  }
  // We initialize the topology if it has not already been done
  if (!m_kernel->isAnOther()) {
    debug() << "\33[1;32m[AlephMatrix::assemble] Initializing topology"
            << "\33[0m";
    ItacRegion(topology->create, AlephMatrix);
    m_kernel->topology()->create(m_setValue_idx);
  }
  // If we have not already calculated the number of non-zero elements per row
  // it is time to trigger it
  debug() << "\33[1;32m[AlephMatrix::assemble] Updating row_nb_element"
          << "\33[0m";
  if (!m_kernel->topology()->hasSetRowNbElements()) {
    UniqueArray<Integer> row_nb_element;
    row_nb_element.resize(m_kernel->topology()->nb_row_rank());
    row_nb_element.fill(0);
    // When we are not an Other, we must update row_nb_element if it was not specified during matrix->create
    if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()) {
      debug() << "\33[1;32m[AlephMatrix::assemble] Kernel's topology has not set its nb_row_elements, now doing it!"
              << "\33[0m";
      const Integer row_offset = m_kernel->topology()->part()[m_kernel->rank()];
      debug() << "\33[1;32m[AlephMatrix::assemble] row_offset=" << row_offset << "\33[0m";
      debug() << "\33[1;32m[AlephMatrix::assemble] filled, row_nb_element.size=" << row_nb_element.size() << "\33[0m";
      // We are doing it in one pass for now to get a maximum bound
      for (Integer i = 0, iMx = m_setValue_row.size(); i < iMx; ++i)
        row_nb_element[m_setValue_row.at(i) - row_offset] += 1;
    }
    m_kernel->topology()->setRowNbElements(row_nb_element);
    debug() << "\33[1;32m[AlephMatrix::assemble] done hasSetRowNbElements"
            << "\33[0m";
  }
  // In the case //, the solver prepares to retrieve matrix parts from others
  debug() << "\33[1;32m[AlephMatrix::assemble] Récupération des parties de matrices"
          << "\33[0m";
  if (m_participating_in_solver && (!m_kernel->configured())) {
    UniqueArray<Integer> nbValues(m_kernel->size());
    {
      ItacRegion(gathered_nb_setValued, AlephMatrix);
      nbValues.fill(0);
      for (Integer iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
        if (m_kernel->rank() != m_ranks[iCpu])
          continue;
        //debug()<<"\33[1;32m[AlephMatrix::assemble] Adding nb_values from iCpu "<<iCpu<<"\33[0m";
        nbValues[iCpu] = m_kernel->topology()->gathered_nb_setValued(iCpu);
      }
    }
    {
      ItacRegion(resize, AlephMatrix);
      m_aleph_matrix_buffer_rows.resize(nbValues);
      m_aleph_matrix_buffer_cols.resize(nbValues);
      m_aleph_matrix_buffer_vals.resize(nbValues);
    }
  }
  // If we are not in //, there is nothing else to do
  if (!m_kernel->isParallel())
    return;
  // If I participate in the solution, I receive contributions from other participants
  if (m_participating_in_solver) {
    ItacRegion(iRecv, AlephMatrix);
    debug() << "\33[1;32m[AlephMatrix::assemble] I am part of the solver, let's iRecv"
            << "\33[0m";
    // If I am the solver, I receive the rest of the matrices from either other cores or myself
    for (Integer iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      // Except from myself
      if (iCpu == m_kernel->rank())
        continue;
      // Except from those who do not participate
      if (m_kernel->rank() != m_ranks[iCpu])
        continue;
      debug() << "\33[1;32m[AlephMatrix::assemble] "
              << " recv " << m_kernel->rank()
              << " <= " << iCpu
              << " size=" << m_aleph_matrix_buffer_cols[iCpu].size() << "\33[0m";
      m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_vals[iCpu], iCpu, false));
      // Once configured, we know all the (i,j): no need to send them back
      if (!m_kernel->configured()) {
        m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_rows[iCpu], iCpu, false));
        m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_cols[iCpu], iCpu, false));
      }
    }
  }
  // If I am an Arcane rank that has data to send, I do it
  if ((m_kernel->rank() != m_ranks[m_kernel->rank()]) && (!m_kernel->isAnOther())) {
    ItacRegion(iSend, AlephMatrix);
    debug() << "\33[1;32m[AlephMatrix::assemble]"
            << " send " << m_kernel->rank()
            << " => " << m_ranks[m_kernel->rank()]
            << " for " << m_setValue_val.size() << "\33[0m";
    m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->send(m_setValue_val, m_ranks[m_kernel->rank()], false));
    if (!m_kernel->configured()) {
      debug() << "\33[1;32m[AlephMatrix::assemble] iSend my row to " << m_ranks[m_kernel->rank()] << "\33[0m";
      m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->send(m_setValue_row, m_ranks[m_kernel->rank()], false));
      debug() << "\33[1;32m[AlephMatrix::assemble] iSend my col to " << m_ranks[m_kernel->rank()] << "\33[0m";
      m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->send(m_setValue_col, m_ranks[m_kernel->rank()], false));
    }
  }
}

/*!
 * \brief create_really transmits the creation order to the external library
 */
void AlephMatrix::
create_really(void)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::create_really");
  debug() << "\33[1;32m[AlephMatrix::create_really]"
          << "\33[0m";
  // We need a working matrix in all cases
  debug() << "\33[1;32m[AlephMatrix::create_really] new MATRIX"
          << "\33[0m";
  // and we trigger the creation within the implementation
  m_implementation->AlephMatrixCreate();
  debug() << "\33[1;32m[AlephMatrix::create_really] done"
          << "\33[0m";
}

/*!
 * \brief assemble_waitAndFill waits for the previously posted requests to be processed
 */
void AlephMatrix::
assemble_waitAndFill(void)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::assemble_waitAndFill");
  debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill]"
          << "\33[0m";
  if (m_kernel->isParallel()) {
    ItacRegion(Wait, AlephMatrix);
    debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] wait for "
            << m_aleph_matrix_mpi_data_requests.size() << " Requests"
            << "\33[0m";
    m_kernel->world()->waitAllRequests(m_aleph_matrix_mpi_data_requests);
    m_aleph_matrix_mpi_data_requests.clear();
    debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] clear"
            << "\33[0m";
    if (m_participating_in_solver == false) {
      debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] nothing more to do"
              << "\33[0m";
    }
  }
  // If I do not participate, I do not participate
  if (!m_participating_in_solver)
    return;
  // Otherwise, we take the time to build the matrix; others should do the same
  if (!m_kernel->configured()) {
    ItacRegion(Create, AlephMatrix);
    debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] solver " << m_index << " create_really"
            << "\33[0m";
    create_really();
  }
  // And then we proceed with filling the matrix
  {
    ItacRegion(Fill, AlephMatrix);
    if (m_kernel->configured())
      m_implementation->AlephMatrixSetFilled(false); // Activation of fill protection
    debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] " << m_index << " fill"
            << "\33[0m";
    AlephInt* bfr_row_implem;
    AlephInt* bfr_col_implem;
    double* bfr_val_implem;
    for (int iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      if (m_kernel->rank() != m_ranks[iCpu])
        continue;
      if (iCpu == m_kernel->rank()) {
        bfr_row_implem = m_setValue_row.data();
        bfr_col_implem = m_setValue_col.data();
        bfr_val_implem = m_setValue_val.data();
        m_implementation->AlephMatrixFill(m_setValue_val.size(),
                                          bfr_row_implem,
                                          bfr_col_implem,
                                          bfr_val_implem);
      }
      else {
        bfr_row_implem = m_aleph_matrix_buffer_rows[iCpu].data();
        bfr_col_implem = m_aleph_matrix_buffer_cols[iCpu].data();
        bfr_val_implem = m_aleph_matrix_buffer_vals[iCpu].data();
        m_implementation->AlephMatrixFill(m_aleph_matrix_buffer_vals[iCpu].size(),
                                          bfr_row_implem,
                                          bfr_col_implem,
                                          bfr_val_implem);
      }
    }
  }
  { // We then declare the matrix as filled, and we start the configuration
    ItacRegion(Cfg, AlephMatrix);
    m_implementation->AlephMatrixSetFilled(true); // Deactivation of fill protection
    if (!m_kernel->configured()) {
      debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] " << m_index << " MATRIX ASSEMBLE"
              << "\33[0m";
      int assrtnd = 0;
      assrtnd = m_implementation->AlephMatrixAssemble();
      debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] AlephMatrixAssemble=" << assrtnd << "\33[0m";
      // throw FatalErrorException("AlephMatrix::assemble_waitAndFill", "configuration failed");
    }
  }
  debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] done"
          << "\33[0m";
}

/*!
  \brief 'Post' the solver to the scheduler asynchronously or not
*/
void AlephMatrix::
solve(AlephVector* x,
      AlephVector* b,
      Integer& nb_iteration,
      Real* residual_norm,
      AlephParams* solver_param,
      bool async)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::solve");
  debug() << "\33[1;32m[AlephMatrix::solve] Queuing solver " << m_index << "\33[0m";
  m_kernel->postSolver(solver_param, this, x, b);
  // If the post was specified to us, we do not trigger synchronous mode
  if (async)
    return;
  debug() << "\33[1;32m[AlephMatrix::solve] SYNCHRONOUS MODE has been requested, syncing!"
          << "\33[0m";
  m_kernel->syncSolver(0, nb_iteration, residual_norm);
  return;
}

/*!
 * \brief Solves the linear system
 * \param x solution of the system Ax=b (output)
 * \param b right-hand side of the system (input)
 * \param nb_iteration number of system iterations (output)
 * \param residual_norm convergence residual of the system (output)
 * \param info parameters of the parallel application (input)
 * \param solver_param Parameters of the solver for the system Ax=b (input)
 */
void AlephMatrix::
solveNow(AlephVector* x,
         AlephVector* b,
         AlephVector* tmp,
         Integer& nb_iteration,
         Real* residual_norm,
         AlephParams* params)
{
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::solveNow");
  const bool dump_to_compare =
  (m_index == 0) && // If we are at the first solution
  (m_kernel->rank() == 0) && // and we are the 'master'
  (params->writeMatrixToFileErrorStrategy()) && // and we requested a write_matrix!
  (m_kernel->subDomain()->commonVariables().globalIteration() == 1) && // and the first or second iteration
  (m_kernel->nbRanksPerSolver() == 1);
  ItacFunction(AlephMatrix);
  if (!m_participating_in_solver) {
    debug() << "\33[1;32m[AlephMatrix::solveNow] Nothing to do here!"
            << "\33[0m";
    return;
  }
  debug() << "\33[1;32m[AlephMatrix::solveNow]"
          << "\33[0m";
  if (dump_to_compare) {
    const Integer globalIteration = m_kernel->subDomain()->commonVariables().globalIteration();
    String mtxFilename = String("m_aleph_matrix_A_") + globalIteration;
    String rhsFilename = String("m_aleph_vector_b_") + globalIteration;
    warning() << "[AlephMatrix::solveNow] mtxFileName rhsFileName write_to_file";
    writeToFile(mtxFilename.localstr());
    b->writeToFile(rhsFilename.localstr());
  }
  // Triggers the solution within the external library
  m_implementation->AlephMatrixSolve(x, b, tmp,
                                     nb_iteration,
                                     residual_norm,
                                     params);
  if (dump_to_compare) {
    const Integer globalIteration = m_kernel->subDomain()->commonVariables().globalIteration();
    String lhsFilename = String("m_aleph_vector_x_") + globalIteration;
    x->writeToFile(lhsFilename.localstr());
  }
  if (m_kernel->isCellOrdering())
    debug() << "\33[1;32m[AlephMatrix::solveSync_waitAndFill] // nb_iteration="
            << nb_iteration << ", residual_norm=" << *residual_norm << "\33[0m";
  if (m_kernel->isParallel())
    return;
  debug() << "\33[1;32m[AlephMatrix::solveSync_waitAndFill] // nb_iteration="
          << nb_iteration << ", residual_norm=" << *residual_norm << "\33[0m";
  return;
}

/*!
 *\brief Triggers the order of retrieving results
 */
void AlephMatrix::
reassemble(Integer& nb_iteration,
           Real* residual_norm)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::reassemble");
  // If we are not in parallel mode, we are done with the solve
  if (!m_kernel->isParallel())
    return;
  m_aleph_matrix_buffer_n_iteration.resize(1);
  m_aleph_matrix_buffer_n_iteration[0] = nb_iteration;
  m_aleph_matrix_buffer_residual_norm.resize(4);
  m_aleph_matrix_buffer_residual_norm[0] = residual_norm[0];
  m_aleph_matrix_buffer_residual_norm[1] = residual_norm[1];
  m_aleph_matrix_buffer_residual_norm[2] = residual_norm[2];
  m_aleph_matrix_buffer_residual_norm[3] = residual_norm[3];
  // We must receive data
  if (m_kernel->rank() != m_ranks[m_kernel->rank()] && !m_kernel->isAnOther()) {
    debug() << "\33[1;32m[AlephMatrix::REassemble] " << m_kernel->rank()
            << "<=" << m_ranks[m_kernel->rank()] << "\33[0m";
    m_aleph_matrix_mpi_results_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_n_iteration,
                                                                    m_ranks[m_kernel->rank()], false));
    m_aleph_matrix_mpi_results_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_residual_norm,
                                                                    m_ranks[m_kernel->rank()], false));
  }
  if (m_participating_in_solver) {
    debug() << "\33[1;32m[AlephMatrix::REassemble] have participated, should send:"
            << "\33[0m";
    for (Integer iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      if (iCpu == m_kernel->rank())
        continue;
      if (m_kernel->rank() != m_ranks[iCpu])
        continue;
      debug() << "\33[1;32m[AlephMatrix::REassemble] " << m_kernel->rank() << " => " << iCpu << "\33[0m";
      m_aleph_matrix_mpi_results_requests.add(m_kernel->world()->send(m_aleph_matrix_buffer_n_iteration,
                                                                      iCpu, false));
      m_aleph_matrix_mpi_results_requests.add(m_kernel->world()->send(m_aleph_matrix_buffer_residual_norm,
                                                                      iCpu, false));
    }
  }
}

/*!
 *\brief Synchronizes the reception of results
 */
void AlephMatrix::
reassemble_waitAndFill(Integer& nb_iteration, Real* residual_norm)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::reassemble_waitAndFill");
  if (!m_kernel->isParallel())
    return;
  debug() << "\33[1;32m[AlephMatrix::REassemble_waitAndFill]"
          << "\33[0m";
  //if (m_kernel->isAnOther()) return;
  m_kernel->world()->waitAllRequests(m_aleph_matrix_mpi_results_requests);
  m_aleph_matrix_mpi_results_requests.clear();
  if (!m_participating_in_solver) {
    nb_iteration = m_aleph_matrix_buffer_n_iteration[0];
    residual_norm[0] = m_aleph_matrix_buffer_residual_norm[0];
    residual_norm[1] = m_aleph_matrix_buffer_residual_norm[1];
    residual_norm[2] = m_aleph_matrix_buffer_residual_norm[2];
    residual_norm[3] = m_aleph_matrix_buffer_residual_norm[3];
  }
  debug() << "\33[1;32m[AlephMatrix::REassemble_waitAndFill] // nb_iteration="
          << nb_iteration << ", residual_norm=" << *residual_norm << "\33[0m";
}

/*!
 *\brief Allows specifying the start of a filling phase
 */
void AlephMatrix::
startFilling()
{
  ItacFunction(AlephMatrix);
  /* Nothing here to do with this m_implementation */
  debug() << "[AlephMatrix::startFilling] void"
          << "\33[0m";
}

/*!
 *\brief Triggers the writing of the matrix to a file
 */
void AlephMatrix::
writeToFile(const String file_name)
{
  ItacFunction(AlephMatrix);
  debug() << "\33[1;32m[AlephMatrix::writeToFile] Dumping matrix to " << file_name << "\33[0m";
  m_implementation->writeToFile(file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
