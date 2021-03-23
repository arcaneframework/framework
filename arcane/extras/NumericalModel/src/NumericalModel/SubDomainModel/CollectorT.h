// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef COLLECTORT_H_
#define COLLECTORT_H_

#include "NumericalModel/Utils/BaseCollector.h"

using namespace Arcane;

class ITimeMng ;
class ITimeIntegrator ;

template<class Model>
class CollectorT : public BaseCollector
{
public :
  typedef VariableCellReal RealVar ;
  typedef std::map<typename Model::RealVarType,RealVar*> RealVarMap ;
  typedef typename RealVarMap::iterator RealVarMapIter ;
  
  CollectorT(ITimeMng* time_mng,
             typename Model::FaceBoundaryConditionMng* bc_mng) 
  : BaseCollector()
  , m_time_mng(time_mng)
  , m_bc_mng(bc_mng)
  , m_time_integrator(NULL)
  , m_use_time_integrator(false)
  {}

  virtual ~CollectorT()
  {
    for(Integer iop=0;iop<m_face_bc_init_ops.size();iop++)
      delete m_face_bc_init_ops[iop] ;
    for(Integer iop=0;iop<m_face_bc_update_ops.size();iop++)
      delete m_face_bc_update_ops[iop] ;
  }

  String getName() const 
  { return "CollectorT" ; }
  
  
  ITimeMng* getTimeMng() {
   return m_time_mng ;
  }
  
  void activateTimeIntegrator(bool flag)
  {
    m_use_time_integrator = flag ;
    if(m_time_integrator==NULL)
      throw Arcane::FatalErrorException(A_FUNCINFO,"Time Integrator is null");
  }
  bool useTimeIntegrator()
  {
    return m_use_time_integrator ;
  }
  
  ITimeIntegrator* getTimeIntegrator()
  {
    return m_time_integrator ;
  }
  void setTimeIntegrator(ITimeIntegrator* time_integrator)
  {
    m_time_integrator = time_integrator ; 
  }
  virtual void startcompute()
  {
    BaseCollector::startcompute() ;
    updateFaceBoundaryCondition() ;
  }
  void addBCInitOp(typename Model::FaceBCOp* op)
  {
    m_face_bc_init_ops.add(op) ;
  }
  void addBCUpdateOp(typename Model::FaceBCOp* op)
  {
    m_face_bc_update_ops.add(op) ;
  }
  void initFaceBoundaryCondition()
  {
    for(Integer iop=0;iop<m_face_bc_init_ops.size();iop++)
      m_face_bc_init_ops[iop]->compute() ;
  }
  void updateFaceBoundaryCondition()
  {
    for(Integer iop=0;iop<m_face_bc_update_ops.size();iop++)
      m_face_bc_update_ops[iop]->compute() ;
  }
  
  void setRealVar(typename Model::RealVarType var_type,RealVar* var)
  {
    m_cell_vars[var_type] = var ;
  }
  RealVar* getRealVar(typename Model::RealVarType var_type)
  {
    RealVarMapIter iter = m_cell_vars.find(var_type) ;
    if(iter==m_cell_vars.end())
      return NULL ;
    else
      return (*iter).second ;
  }
private :
  ITimeMng* m_time_mng ;
  typename Model::FaceBoundaryConditionMng* m_bc_mng ;
  ITimeIntegrator* m_time_integrator ;
  
  bool m_use_time_integrator ;
  
  BufferT< typename Model::FaceBCOp* > m_face_bc_init_ops ;
  BufferT< typename Model::FaceBCOp* > m_face_bc_update_ops ;
  
  RealVarMap m_cell_vars ;
};

#endif /*COLLECTORT_H_*/
