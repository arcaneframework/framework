// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef TIMEINTEGRATOR_H_
#define TIMEINTEGRATOR_H_

#include "NumericalModel/Algorithms/ITimeIntegrator.h"
#include "NumericalModel/Algorithms/Integrator/IntegratorT.h"

using namespace Arcane;

class INumericalModel ;
class ITimeMng ;
class ITimeStepMng ;

class TimeStepIterator
{
public :
  
  typedef ITimeIntegrator::eErrorType eErrorType ;
  
  TimeStepIterator(INumericalModel* model) ;
  
  TimeStepIterator& operator++() ;
  bool stop() {
    return m_stop ;
  }
  bool isOk() {
    return m_error==0 ;
  }
  
  eErrorType getError() 
  {
    return m_error ;
  }
  static eErrorType initError() {
    return ITimeIntegrator::NoError ;
  }
private :
  ITimeMng* m_time_mng ;
  ITimeStepMng* m_time_step_mng ;
  Real m_init_time ;
  Real m_final_time ;
  Real m_last_time ;
  Real m_current_time ;
  Real m_current_deltat ;
  bool m_stop ;
  eErrorType m_error ;
} ;

class TimeIntegrator
: public ITimeIntegrator
, public IntegratorT<TimeStepIterator>
{
public :
  TimeIntegrator() ;

  virtual ~TimeIntegrator() {}

  bool integrate(INumericalModel* model, Integer sequence) {
    return IntegratorT<TimeStepIterator>::integrate(model,sequence) ;
  }
  
  ITimeIntegrator::eErrorType getError() const 
  {
    return IntegratorT<TimeStepIterator>::getError() ;
  }
};
#endif /*TIMEINTEGRATOR_H_*/
