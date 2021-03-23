// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef INTEGRATORT_H_
#define INTEGRATORT_H_

#include "NumericalModel/Algorithms/IIntegrator.h"
#include "NumericalModel/Models/INumericalModel.h"

using namespace Arcane;

class INumericalModel;

template<typename Iter>
class IntegratorT : public IIntegrator
{
public :
  IntegratorT()
  : m_error(Iter::initError())
  {}

  bool integrate(INumericalModel * model, Integer sequence)
  {
    model->start(sequence) ;
    Iter iter(model) ;
    while (!iter.stop())
    {
      model->baseCompute(sequence);
      ++iter ;
    }
    model->finalize(sequence) ;
    m_error = iter.getError() ;
    return iter.isOk() ;
  }
  
  typename Iter::eErrorType getError() const
  {
    return m_error ;
  }
protected :
  typename Iter::eErrorType m_error ;
} ;

#endif
