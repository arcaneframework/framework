// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef COUPLINGMODELSOLVERT_H_
#define COUPLINGMODELSOLVERT_H_

#include "NumericalModel/Algorithms/ICouplingModelSolver.h"

using namespace Arcane;

class IIntOp;

template<class Model>
class SolverIterT
{
public :
  typedef enum
  {
    NoError,
    MaxIterError,
    ModelError
  } eErrorType ;
  SolverIterT(Model* model,
             IIntOp* stop_criteria,
             Integer max_iter)
  : m_parent(model)
  , m_iter(0)
  , m_max_iter(max_iter)
  , m_error(NoError)
  , m_stop_criteria(stop_criteria)
  {
  }
  SolverIterT<Model>& operator++()
  {
    if(m_parent->getError()!=0)
      m_error=ModelError ;
    m_iter++;
    return *this;
  }
  bool isOk() {
    return m_error==NoError ;
  }
  bool stop()
  {
    if(m_iter==m_max_iter)
    {
      m_error = MaxIterError ;
      return true ;
    }
    if (m_iter > 0)
      return (m_stop_criteria->compute()==0) ;
    return false ;
  }
  Integer value() 
  {
    return m_iter ;
  }
private :
  Model* m_parent ;
  Integer m_iter ;
  Integer m_max_iter ;
  eErrorType m_error ;
  IIntOp* m_stop_criteria ;
} ;

class INumericalModel;
class IOp;

template<class Model>
class CouplingModelSolverT : public ICouplingModelSolver
{
public :
   CouplingModelSolverT(Model* model, 
                        IIntOp* stop_criteria_op,
                        IOp* update_op,
                        Integer max_iter)
   : m_parent(model)
   , m_stop_criteria_op(stop_criteria_op)
   , m_update_op(update_op)
   , m_max_iter(max_iter)
   , m_verbose(false)
   {}
   virtual ~CouplingModelSolverT() {}
   bool solve(INumericalModel* model1, Integer seq1,
              INumericalModel* model2, Integer seq2)
   {
     if(m_verbose)
     {
       m_parent->info();
       m_parent->info();
       m_parent->info() << "################################" ;
       m_parent->info() << "# START COUPLING SOLVER" ;
       m_parent->info() << "# Model 1 :"<<model1->getName();
       m_parent->info() << "# Model 2 :"<<model2->getName();
       m_parent->info() << "################################" ;
       m_parent->info();
     }
     SolverIterT<Model> iter(m_parent,m_stop_criteria_op,m_max_iter) ;
     while(!iter.stop())
     {
       if(m_verbose)
       {
         m_parent->info() << "#############################################" ;
         m_parent->info() << "# COUPLING ITERATIONS "<< iter.value() ;
         m_parent->info() << "#############################################" ;
       }
       model1->compute(seq1);
       model2->compute(seq2);
       //m_time_integrator->integrate(fine_dt_model,FineDtSeq) ;
       ++iter ;
     }
     if(iter.isOk())
     {
       if(m_verbose)
       {
         m_parent->info();
         m_parent->info() << "#############################################" ;
         m_parent->info() << "# END OF COUPLING SOLVER" ;
         m_parent->info() << "# Convergence in "<<iter.value()<< " iterations";
         m_parent->info() << "#############################################" ;
         m_parent->info();
       }
       if(m_update_op)
         m_update_op->compute() ;
       //m_parent->updatePressure() ;
     }
     else
     {
       if(m_verbose)
       {
         m_parent->info() << "#############################################" ;
         m_parent->info() << "# END OF COUPLING SOLVER WITH ERRORS" ;
         m_parent->info() << "#############################################" ;
       }
     }
     return iter.isOk() ;
   }
   void setVerbose(bool flag)
   {
     m_verbose = flag ;
   }
private :
   Model* m_parent ;
   IIntOp* m_stop_criteria_op ;
   IOp* m_update_op ;
   Integer m_max_iter ;
   bool m_verbose ;
} ;


#endif /*COUPLINGMODELSOLVERT_H_*/
