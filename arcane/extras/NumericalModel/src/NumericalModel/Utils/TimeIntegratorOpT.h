// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef TIMEINTEGRATOROPT_H
#define TIMEINTEGRATOROPT_H

#include "NumericalModel/Algorithms/ITimeIntegrator.h"

using namespace Arcane;

template<class Model>
class TimeIntegratorOpT : public Model::Visitor
{
public:
  TimeIntegratorOpT(Integer sequence,ITimeIntegrator* integrator)
  : m_sequence(sequence)
  , m_time_integrator(integrator)
  {
  }
  virtual ~TimeIntegratorOpT()
  {
  }
  String getName() { return "TimeIntegratorOpT" ; }

  virtual Integer visit(Model* model)
  {
    if(m_time_integrator->integrate(model,m_sequence))
      return 0 ;
    else
      return 1 ;
  }
  virtual Integer visitForStart(Model* model)
  { return 0 ; }
  virtual Integer visitForFinalize(Model* model)
  { return 0 ; }
private :
  Integer m_sequence ;
  ITimeIntegrator* m_time_integrator ;
};

#endif

