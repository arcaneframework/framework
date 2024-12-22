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
  // Récupération des rangs utilisés pour cette résolution
  m_ranks = kernel->solverRanks(m_index);
  // Booléen pour savoir si on participe ou pas
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
  // On va chercher une matrice depuis la factory qui fait l'interface aux bibliothèques externes
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
 * Matrix 'create' avec l'API 'void'
 * BaseForm[Hash["AlephMatrix::create(void)", "CRC32"], 16] = fff06e2
 *****************************************************************************/
void AlephMatrix::create(void)
{
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::create");
  debug() << "\33[1;32m[AlephMatrix::create(void)]\33[0m";
  // Si le kernel n'est pas initialisé, on a rien à faire
  if (!m_kernel->isInitialized())
    return;
  // S'il y a des 'autres' et qu'on en fait pas partie,
  // on broadcast qu'un 'create' est à faire
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther())
    m_kernel->world()->broadcast(UniqueArray<unsigned long>(1, 0xfff06e2l).view(), 0);
  // On flush en prévision du remplissage, il faut le faire même étant configuré
  // Par contre, on ne flush pas celui du addValue
  m_setValue_idx = 0;
}

/******************************************************************************
 * Matrix 'create' avec l'API qui spécifie le nombre d'éléments non nuls pas lignes
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
 * \brief reset pour flusher les tableaux des [set&add]Value
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
 * \brief addValue à partir d'arguments en IVariables, Items et Real
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
  // On fait de même coté 'set' pour avoir la bonne taille
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
  // Si la row n'est même pas encore connue
  // On rajoute une entrée map(map(m_addValue_idx))
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
  // On focus sur la seconde dimension
  colMap* jMap = iRowMap->second;
  colMap::const_iterator iColMap = jMap->find(col);
  // Si cet col n'est pas connue de cette row
  // On rajoute une entrée
  if (iColMap == jMap->end()) {
    /*debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] col "
          <<col<<" inconue, m_addValue_idx="
          <<m_addValue_idx<<"\33[0m";*/
    jMap->insert(std::make_pair(col, m_addValue_idx));
    updateKnownRowCol(row, col, val);
    return;
  }
  // Sinon on ajoute
  //debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] hit\33[0m";
  //debug()<<"[AlephMatrix::rowMapMapCol] += for ["<<row<<","<<col<<"]="<<val; traceMng()->flush();
  m_addValue_val[iColMap->second] += val;
}

/*!
 * \brief addValue standard en (i,j,val)
 */
void AlephMatrix::
addValue(Integer row, Integer col, Real val)
{
  //debug()<<"\33[32m[AlephMatrix::addValue] addValue("<<row<<","<<col<<")="<<val<<"\33[0m";
  row = m_kernel->ordering()->swap(row);
  col = m_kernel->ordering()->swap(col);
  // Recherche de la case (row,j) si elle existe déjà
  rowMapMapCol(row, col, val);
}

/*!
 * \brief setValue à partir d'arguments en IVariables, ItemEnumerator et Real
 */
void AlephMatrix::
setValue(const VariableRef& rowVar, const ItemEnumerator& rowItm,
         const VariableRef& colVar, const ItemEnumerator& colItm,
         const Real val)
{
  setValue(rowVar, *rowItm, colVar, *colItm, val);
}

/*!
 * \brief setValue à partir d'arguments en IVariables, Items et Real
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
 * \brief setValue standard à partir d'arguments (row,col,val)
 */
void AlephMatrix::
setValue(Integer row, Integer col, Real val)
{
  // Re-ordering si besoin
  row = m_kernel->ordering()->swap(row);
  col = m_kernel->ordering()->swap(col);
  // Si le kernel a déjà été configuré,
  // on s'assure que la 'géométrie/support' n'a pas changée entre les résolutions
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
 * \brief reIdx recherche la correspondance de l'AlephIndexing
 */
Int32 AlephMatrix::
reIdx(Integer ij,
      Array<Int32*>& known_items_own_address)
{
  return *known_items_own_address[ij];
}

/*!
 * \brief reSetValuesIn rejoue les setValue avec les indexes calculés via l'AlephIndexing
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
 * \brief reAddValuesIn rejoue les addValue avec les indexes calculés via l'AlephIndexing
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
 * \brief assemble les matrices avant résolution
 */
void AlephMatrix::
assemble(void)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::assemble");
  // Si le kernel n'est pas initialisé, on ne fait toujours rien
  if (!m_kernel->isInitialized()) {
    debug() << "\33[1;32m[AlephMatrix::assemble] Trying to assemble a matrix"
            << "from an uninitialized kernel!\33[0m";
    return;
  }
  // Si aucun [set|add]Value n'a été perçu, ce n'est pas normal
  if (m_addValue_idx != 0 && m_setValue_idx != 0)
    throw FatalErrorException("AlephMatrix::assemble", "Still exclusives [add||set]Value required!");
  // Si des addValue ont été captés, il faut les 'rejouer'
  // Attention: pour l'instant les add et les set sont disjoints!
  if (m_addValue_idx != 0) {
    debug() << "\33[1;32m[AlephMatrix::assemble] m_addValue_idx!=0\33[0m";
    // On flush notre index des setValues
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
  // S'il y a des 'autres' et qu'on en fait pas parti, on les informe de l'assemblage
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()) {
    debug() << "\33[1;32m[AlephMatrix::assemble] On informe les autres kappa que l'on assemble"
            << "\33[0m";
    m_kernel->world()->broadcast(UniqueArray<unsigned long>(1, 0x74f253cal).view(), 0);
    // Et on leur donne l'info du m_setValue_idx
    m_kernel->world()->broadcast(UniqueArray<Integer>(1, m_setValue_idx).view(), 0);
  }
  // On initialise la topologie si cela n'a pas été déjà fait
  if (!m_kernel->isAnOther()) {
    debug() << "\33[1;32m[AlephMatrix::assemble] Initializing topology"
            << "\33[0m";
    ItacRegion(topology->create, AlephMatrix);
    m_kernel->topology()->create(m_setValue_idx);
  }
  // Si on a pas déjà calculé le nombre d'éléments non nuls par lignes
  // c'est le moment de le déclencher
  debug() << "\33[1;32m[AlephMatrix::assemble] Updating row_nb_element"
          << "\33[0m";
  if (!m_kernel->topology()->hasSetRowNbElements()) {
    UniqueArray<Integer> row_nb_element;
    row_nb_element.resize(m_kernel->topology()->nb_row_rank());
    row_nb_element.fill(0);
    // Quand on est pas un Autre, il faut mettre à jour le row_nb_element si cela n'a pas été spécifié lors du matrice->create
    if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()) {
      debug() << "\33[1;32m[AlephMatrix::assemble] Kernel's topology has not set its nb_row_elements, now doing it!"
              << "\33[0m";
      const Integer row_offset = m_kernel->topology()->part()[m_kernel->rank()];
      debug() << "\33[1;32m[AlephMatrix::assemble] row_offset=" << row_offset << "\33[0m";
      debug() << "\33[1;32m[AlephMatrix::assemble] filled, row_nb_element.size=" << row_nb_element.size() << "\33[0m";
      // On le fait pour l'instant en une passe pour avoir une borne max
      for (Integer i = 0, iMx = m_setValue_row.size(); i < iMx; ++i)
        row_nb_element[m_setValue_row.at(i) - row_offset] += 1;
    }
    m_kernel->topology()->setRowNbElements(row_nb_element);
    debug() << "\33[1;32m[AlephMatrix::assemble] done hasSetRowNbElements"
            << "\33[0m";
  }
  // Dans le cas //, le solveur se prépare à récupérer les parties de matrices venant des autres
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
  // Si on est pas en //, on a rien d'autre à faire
  if (!m_kernel->isParallel())
    return;
  // Si je participe à la résolution, je reçois les contributions des autres participants
  if (m_participating_in_solver) {
    ItacRegion(iRecv, AlephMatrix);
    debug() << "\33[1;32m[AlephMatrix::assemble] I am part of the solver, let's iRecv"
            << "\33[0m";
    // Si je suis le solveur, je recv le reste des matrices provenant soit des autres coeurs, soit de moi-même
    for (Integer iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      // Sauf de moi-même
      if (iCpu == m_kernel->rank())
        continue;
      // Sauf de ceux qui ne participent pas
      if (m_kernel->rank() != m_ranks[iCpu])
        continue;
      debug() << "\33[1;32m[AlephMatrix::assemble] "
              << " recv " << m_kernel->rank()
              << " <= " << iCpu
              << " size=" << m_aleph_matrix_buffer_cols[iCpu].size() << "\33[0m";
      m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_vals[iCpu], iCpu, false));
      // Une fois configuré, nous connaissons tous les (i,j): pas besoin de les renvoyer
      if (!m_kernel->configured()) {
        m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_rows[iCpu], iCpu, false));
        m_aleph_matrix_mpi_data_requests.add(m_kernel->world()->recv(m_aleph_matrix_buffer_cols[iCpu], iCpu, false));
      }
    }
  }
  // Si je suis un rang Arcane qui a des données à envoyer, je le fais
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
 * \brief create_really transmet l'ordre de création à la bibliothèque externe
 */
void AlephMatrix::
create_really(void)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::create_really");
  debug() << "\33[1;32m[AlephMatrix::create_really]"
          << "\33[0m";
  // Il nous faut alors dans tous les cas une matrice de travail
  debug() << "\33[1;32m[AlephMatrix::create_really] new MATRIX"
          << "\33[0m";
  // et on déclenche la création au sein de l'implémentation
  m_implementation->AlephMatrixCreate();
  debug() << "\33[1;32m[AlephMatrix::create_really] done"
          << "\33[0m";
}

/*!
 * \brief assemble_waitAndFill attend que les requètes précédemment postées aient été traitées
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
  // Si je ne participe pas, je ne participe pas
  if (!m_participating_in_solver)
    return;
  // Sinon, on prend le temps de construire la matrice, les autres devraient le faire aussi
  if (!m_kernel->configured()) {
    ItacRegion(Create, AlephMatrix);
    debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] solver " << m_index << " create_really"
            << "\33[0m";
    create_really();
  }
  // Et on enchaîne alors avec le remplissage de la matrice
  {
    ItacRegion(Fill, AlephMatrix);
    if (m_kernel->configured())
      m_implementation->AlephMatrixSetFilled(false); // Activation de la protection de remplissage
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
  { // On déclare alors la matrice comme remplie, et on lance la configuration
    ItacRegion(Cfg, AlephMatrix);
    m_implementation->AlephMatrixSetFilled(true); // Désactivation de la protection de remplissage
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
  \brief 'Poste' le solver au scheduler de façon asynchrone ou pas
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
  // Si on nous a spécifié le post, on ne déclenche pas le mode synchrone
  if (async)
    return;
  debug() << "\33[1;32m[AlephMatrix::solve] SYNCHRONOUS MODE has been requested, syncing!"
          << "\33[0m";
  m_kernel->syncSolver(0, nb_iteration, residual_norm);
  return;
}

/*!
 * \brief Résout le système linéraire
 * \param x solution du système Ax=b (en sortie)
 * \param b second membre du système (en entrée)
 * \param nb_iteration nombre d'itérations du système (en sortie)
 * \param residual_norm résidu de convergence du système (en sortie)
 * \param info parametres de l'application parallele (en entrée)
 * \param solver_param Parametres du Solveur du solveur Ax=b (en entrée)
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
  (m_index == 0) && // Si on est à la première résolution
  (m_kernel->rank() == 0) && // et qu'on est le 'master'
  (params->writeMatrixToFileErrorStrategy()) && // et qu'on a demandé un write_matrix !
  (m_kernel->subDomain()->commonVariables().globalIteration() == 1) && // et la première itération ou la deuxième
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
  // Déclenche la résolution au sein de la bibliothèque externe
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
 *\brief Déclenche l'ordre de récupération des résultats
 */
void AlephMatrix::
reassemble(Integer& nb_iteration,
           Real* residual_norm)
{
  ItacFunction(AlephMatrix);
  Timer::Action ta(m_kernel->subDomain(), "AlephMatrix::reassemble");
  // Si on est pas en mode parallèle, on en a finit pour le solve
  if (!m_kernel->isParallel())
    return;
  m_aleph_matrix_buffer_n_iteration.resize(1);
  m_aleph_matrix_buffer_n_iteration[0] = nb_iteration;
  m_aleph_matrix_buffer_residual_norm.resize(4);
  m_aleph_matrix_buffer_residual_norm[0] = residual_norm[0];
  m_aleph_matrix_buffer_residual_norm[1] = residual_norm[1];
  m_aleph_matrix_buffer_residual_norm[2] = residual_norm[2];
  m_aleph_matrix_buffer_residual_norm[3] = residual_norm[3];
  // Il faut recevoir des données
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
 *\brief Synchronise les réceptions des résultats
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
 *\brief Permet de spécifier le début d'une phase de remplissage
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
 *\brief Déclenche l'écriture de la matrice dans un fichier
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
