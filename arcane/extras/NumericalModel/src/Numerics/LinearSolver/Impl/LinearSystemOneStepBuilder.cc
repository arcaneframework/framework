// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Numerics/LinearSolver/Impl/LinearSystemOneStepBuilder.h"

#include "Utils/Utils.h"

#include "Numerics/LinearSolver/ILinearSystem.h"
#include "Numerics/LinearSolver/Impl/BasicIndexManagerImpl.h"
#include "Numerics/LinearSolver/HypreSolverImpl/HypreLinearSystem.h"

LinearSystemOneStepBuilder::
LinearSystemOneStepBuilder() :
  ILinearSystemBuilder(),
  m_state(eNone),
  m_solver_type(eUndefined),
  m_trace(NULL),
  m_parallel_mng(NULL),
  m_index_manager(NULL)
{
  ;
}

/*---------------------------------------------------------------------------*/

LinearSystemOneStepBuilder::
~LinearSystemOneStepBuilder()
{
  delete m_index_manager;
}

/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
addData(const Integer iIndex,
	const Integer jIndex,
	const Real value)
{ 
  ARCANE_ASSERT((m_state == eStart),("Unexpected state: %d vs %d",m_state,eStart));
  m_matrix[iIndex][jIndex] += value;
}
  
/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
addData(const Integer iIndex,
	const Real factor,
	const ConstArrayView<Integer> & jIndexes,
	const ConstArrayView<Real> & jValues)
{
  ARCANE_ASSERT((m_state == eStart),("Unexpected state: %d vs %d",m_state,eStart));
  ARCANE_ASSERT((jIndexes.size() == jValues.size()),("Inconsistent sizes: %d vs %d",jIndexes.size(),jValues.size()));
  const Integer n = jIndexes.size();
  for(Integer j=0;j<n;++j)
    {
      m_matrix[iIndex][jIndexes[j]] += factor * jValues[j];
    }
}


/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
addRHSData(const Integer iIndex,
	   const Real value) 
{
  ARCANE_ASSERT((m_state == eStart),("Unexpected state: %d vs %d",m_state,eStart));
  m_rhs_vector[iIndex] += value;
}



/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
setRHSData(const Integer iIndex,
	   const Real value)
{ 
  ARCANE_ASSERT((m_state == eStart),("Unexpected state: %d vs %d",m_state,eStart));
  m_rhs_vector[iIndex] = value;
}

/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
init() 
{ 
  if (m_trace == NULL)
    throw FatalErrorException("Undefined Trace Manager");

  static int si = 0;
  m_trace->debug() << "Initialisation : " << ++si;
  
  if (m_parallel_mng == NULL)
    m_trace->fatal() << "Undefined Parallel Manager";

  if (m_index_manager) delete m_index_manager;
  m_index_manager = new BasicIndexManagerImpl(m_parallel_mng);
  m_index_manager->setTraceMng(m_trace);
  m_index_manager->init();

  m_matrix.clear();
  m_rhs_vector.clear();
  m_state = eInit;
}
/*---------------------------------------------------------------------------*/
void 
LinearSystemOneStepBuilder::
initRhs()
{   
  /*if (m_index_manager) delete m_index_manager;
  m_index_manager = new BasicIndexManagerImpl(m_parallel_mng);
  m_index_manager->setTraceMng(m_trace);
  m_index_manager->init();
  */
  m_rhs_vector.clear();
  m_state = eStart;
} 
/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
start() 
{ 
  ARCANE_ASSERT((m_state == eInit),("Unexpected state: %d vs %d",m_state,eInit));
  m_index_manager->prepare();
  m_state = eStart;
}

/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
end() 
{ 
  m_state = eInit;
}

/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
freeData() 
{
#warning "TODO: FreeData à implementer"
}

/*---------------------------------------------------------------------------*/

IIndexManager * 
LinearSystemOneStepBuilder::
getIndexManager() 
{ 
  return m_index_manager;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool 
LinearSystemOneStepBuilder::
build(HypreLinearSystem * system) 
{
#ifdef USE_HYPRE
  ARCANE_ASSERT((m_state == eStart),("Unexpected state: %d vs %d",m_state,eStart));

  // on_the_fly_destroy destroys values immediatly after copy, 
  // do not wait the end of this build process
  bool on_the_fly_destroy = false;

  ARCANE_ASSERT((system != NULL),("Unexpected non null system"));
  
  Integer data_count = 0;
  Integer pos = 0;
  BufferT<int> sizes(m_matrix.size());
  for(MatrixData::const_iterator i=m_matrix.begin(); i!=m_matrix.end(); ++i) {
    data_count += i->second.size();
    sizes[pos] = i->second.size();
    ++pos;
  }
  
  // Synthèse de l'indexation
  Integer global_size, local_offset, local_size;
  m_index_manager->stats(global_size, local_offset, local_size);
  
  if (local_size != (Integer)m_matrix.size())
    m_trace->fatal() << "Inconsistent system size : equation support are local items only! (" << local_size << " vs " << m_matrix.size() << ")";
    
  int ilower = local_offset;
  int iupper = local_offset+local_size-1;
  int jlower = ilower;
  int jupper = iupper;
  
  m_trace->debug() << "Matrix range : " 
		   << "[" << ilower << ":" << iupper << "]" << "x"
		   << "[" << jlower << ":" << jupper << "]";

  if (not system->initMatrix(ilower,iupper,
			     jlower,jupper,
			     sizes))
    {
      m_trace->error() << "Hypre Initialisation failed";
      return false ;
    }
    
  // Buffer de construction
  Integer jpos;
  BufferT<double> values(local_size);
  BufferT<int> & indices = sizes; // réutilisation du buffer
    
  for(MatrixData::iterator i=m_matrix.begin(); i!=m_matrix.end(); ++i) {
    VectorData & line = i->second;
    int row = i->first;
    int ncols = line.size();
    jpos = 0;
    for(VectorData::const_iterator j=line.begin(); j != line.end(); ++j)
      {
	indices[jpos] = j->first;
	values[jpos] = j->second;
	++jpos;
      }
      
    if (not system->setMatrixValues(1,&row,&ncols,indices.unguardedBasePointer(),
				    values.unguardedBasePointer()))
      {
        m_trace->error() << "Cannot set Hypre Matrix Values for row " << row;
        return false ;
      }

    if (on_the_fly_destroy) line.clear();
  }

  // -- B Vector --
  jpos = 0;
  for(VectorData::const_iterator j=m_rhs_vector.begin(); j != m_rhs_vector.end(); ++j)
    {
      indices[jpos] = j->first;
      values[jpos] = j->second;
      ++jpos;
    }
  if (on_the_fly_destroy) m_rhs_vector.clear();

  if (not system->setRHSValues(jpos, // nb de valeurs
			       indices.unguardedBasePointer(),
			       values.unguardedBasePointer()))
    {
      m_trace->error() << "Hypre RHS Vector Initilization error";
      return false ;
    }
      

  initUnknown(system);
      
  if (not system->assemble())
    {
      m_trace->error() << "Hypre assembling failed"; 
      return false ;
    }
  return true ;
#else
  m_trace->fatal()<<"HYPRE SOLVER not provided";
  return false ;
#endif
}

/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
initUnknown(HypreLinearSystem * system)
{
#ifdef USE_HYPRE
  return initUnknownT(system);
#endif
}

/*---------------------------------------------------------------------------*/

bool 
LinearSystemOneStepBuilder::
commitSolution(HypreLinearSystem * system)
{
#ifdef USE_HYPRE
  return commitSolutionT(system);
#else
  return false ;
#endif
}

/*---------------------------------------------------------------------------*/

bool 
LinearSystemOneStepBuilder::
connect(HypreLinearSystem * system)
{
#ifdef USE_HYPRE
  m_solver_type = eHypre;
  ISubDomain* sub_domain = system->getSubDomain() ;
  m_trace = sub_domain->traceMng();
  m_parallel_mng = sub_domain->parallelMng() ;
  return true;
#else
  m_trace->fatal()<<"HYPRE SOLVER not provided";
  return false ;
#endif
}

/*---------------------------------------------------------------------------*/

bool 
LinearSystemOneStepBuilder::
visit(HypreLinearSystem * system)
{
  return build(system) ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SystemT>
void 
LinearSystemOneStepBuilder::
initUnknownT(SystemT * system)
{
  ARCANE_ASSERT((system != NULL),("Unexpected non null system"));
  
  // Un container pour l'ensemble alloué à la taille max
  Array<Real> values;
  Integer max_size = 0;
  for(IIndexManager::EntryEnumerator i = m_index_manager->enumerateEntry(); i.hasNext(); ++i)
    if (i->needInit())
      max_size = math::max(max_size,i->getIndex().size());
  values.reserve(max_size);

  for(IIndexManager::EntryEnumerator i = m_index_manager->enumerateEntry(); i.hasNext(); ++i)
    {
      if (i->needInit())
        {
          m_trace->debug() << "Initializing Entry : " << i->getName();
          
	        ConstArrayView<Integer> indices = i->getIndex();
          if (not indices.empty()) 
            {
              values.resize(indices.size());
              i->initValues(values) ;
              
              if (not system->setInitValues(indices.size(),
                                            indices.unguardedBasePointer(),
                                            values.unguardedBasePointer()))
                m_trace->fatal() << system->name() << " Initial Vector Initialization error";
            }
        }
    }
}

/*---------------------------------------------------------------------------*/

template<typename SystemT>
bool 
LinearSystemOneStepBuilder::
commitSolutionT(SystemT * system)
{
  ARCANE_ASSERT((system != NULL),("Unexpected non null system"));
  
  // Un container pour l'ensemble alloué à la taille max
  Array<Real> values;
  Integer max_size = 0;
  for(IIndexManager::EntryEnumerator i = m_index_manager->enumerateEntry(); i.hasNext(); ++i)
    if (i->needInit())
      max_size = math::max(max_size,i->getIndex().size());
  values.reserve(max_size);
  
  for(IIndexManager::EntryEnumerator i = m_index_manager->enumerateEntry(); i.hasNext(); ++i)
    {
      if (i->needUpdate()) 
        {
          m_trace->debug() << "Updating Entry : " << i->getName();
          
          ConstArrayView<Integer> indices = i->getIndex();
          if (not indices.empty())
            {
              values.resize(indices.size());

              if (not system->getSolutionValues(indices.size(),
                                                indices.unguardedBasePointer(),
                                                values.unguardedBasePointer()))
                {
                  m_trace->fatal() << system->name() << " Solution Vector reading error";
                  return false ;
                }

              i->updateValues(values);
            }
        }
    }
  return true ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
LinearSystemOneStepBuilder::
dumpToMatlab(const std::string& file_name) 
{
  ARCANE_ASSERT((m_state == eStart),("Unexpected state: %d vs %d",m_state,eStart));

  std::ofstream output(file_name.c_str(), std::ios::out);  

  // Dump matrix
  output << "A = [..." << std::endl;
  for(MatrixData::const_iterator i = m_matrix.begin(); i != m_matrix.end(); i++)
    for(VectorData::const_iterator j = i->second.begin(); j != i->second.end(); j++)
      output << i->first + 1 << ", " << j->first + 1 << ", " 
             << j->second << ";..." << std::endl;
  output << "];" << std::endl;

  // Dump rhs vector
  output << "b = [..." << std::endl;
  for(VectorData::const_iterator i = m_rhs_vector.begin(); i != m_rhs_vector.end(); i++)
    output << i->first + 1 << ", " << i->second << ";..." << std::endl;
  output << "];" << std::endl;
}

/*---------------------------------------------------------------------------*/

