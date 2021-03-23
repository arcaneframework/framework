// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef BASECOLLECTOR_H
#define BASECOLLECTOR_H

#include "NumericalModel/Utils/ICollector.h"
#include "NumericalModel/Utils/IOp.h"

using namespace Arcane;

class BaseCollector : public ICollector
{
public :
  BaseCollector() {}
  virtual ~BaseCollector() 
  {
    for(Integer iop=0;iop<m_start_ops.size();iop++)
      delete m_start_ops[iop] ;
    for(Integer iop=0;iop<m_compute_ops.size();iop++)
      delete m_startcompute_ops[iop] ;
    for(Integer iop=0;iop<m_compute_ops.size();iop++)
      delete m_compute_ops[iop] ;
    for(Integer iop=0;iop<m_compute_ops.size();iop++)
      delete m_finalizecompute_ops[iop] ;
    for(Integer iop=0;iop<m_finalize_ops.size();iop++)
      delete m_finalize_ops[iop] ;
  }
  virtual String getName() const 
  {
     return "BaseCollector" ;
  }
  
  //record operators and manage is life
  void addOperator(ICollector::eActionType type,IOp* op)
  {
     switch(type)
     {
        case ICollector::Start :
           m_start_ops.add(op) ;
           break ;
        case ICollector::StartCompute :
           m_startcompute_ops.add(op) ;
           break ;
        case ICollector::Compute :
           m_compute_ops.add(op) ;
           break ;
        case ICollector::FinalizeCompute :
           m_finalizecompute_ops.add(op) ;
           break ;
        case ICollector::Finalize :
           m_finalize_ops.add(op) ;
           break ;
     }
  }

  virtual void start()
  {
     for(Integer iop=0;iop<m_start_ops.size();iop++)
        m_start_ops[iop]->compute() ;
  }
  
  virtual void startcompute()
  {
     for(Integer iop=0;iop<m_startcompute_ops.size();iop++)
        m_startcompute_ops[iop]->compute() ;
  }
  virtual void compute()
  {
     for(Integer iop=0;iop<m_compute_ops.size();iop++)
        m_compute_ops[iop]->compute() ;
  }
  virtual void finalizecompute()
  {
     for(Integer iop=0;iop<m_finalizecompute_ops.size();iop++)
        m_finalizecompute_ops[iop]->compute() ;
  }
  virtual void finalize()
  {
     for(Integer iop=0;iop<m_finalize_ops.size();iop++)
        m_finalize_ops[iop]->compute() ;
  }
  void compute(ICollector::eActionType type)
  {
     switch(type)
     {
        case ICollector::Start :
           start() ;
           break ;
        case ICollector::StartCompute :
           startcompute() ;
           break ;
        case ICollector::Compute :
           compute() ;
           break ;
        case ICollector::FinalizeCompute :
           finalizecompute() ;
           break ;
        case ICollector::Finalize :
           finalize() ;
           break ;
     }
  }
  
protected :
  BufferT<IOp*> m_start_ops ;
  BufferT<IOp*> m_startcompute_ops ;
  BufferT<IOp*> m_compute_ops ;
  BufferT<IOp*> m_finalizecompute_ops ;
  BufferT<IOp*> m_finalize_ops ;
};
#endif
