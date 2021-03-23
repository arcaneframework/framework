// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef TIMESTEPMODELOPT_H
#define TIMESTEPMODELOPT_H

class ITimeMng ;
class ICollector ;
template<class Model>
class TimeStepModelOpT : public Model::Visitor
{
public:
  TimeStepModelOpT(ITimeMng* time_mng,ICollector* collector)
  : m_time_mng(time_mng)
  , m_parent(collector)
  {
  }
  virtual ~TimeStepModelOpT()
  {
  }
  String getName() { return "TimeStepModelOpT" ; }

  virtual Integer visit(Model* model)
  {
    if(m_parent)
    {
      m_parent->startcompute() ;
      m_parent->compute() ;
    }
    model->startTimeStep() ;
    model->baseCompute() ;
    model->endTimeStep() ;
    Integer error =  model->getError() ;
    if((error==0)&&(m_parent))
         m_parent->finalizecompute() ;
    return error ;
  }
  virtual Integer visitForStart(Model* model)
  {
    model->setTimeMng(m_time_mng) ;
    model->setVars(m_parent) ;
    if(m_parent)
      m_parent->start() ;
    return model->getError() ;
  }
  virtual Integer visitForFinalize(Model* model)
  {
    model->setTimeMng(NULL) ;
    if(m_parent)
      m_parent->finalize() ;
    return model->getError() ;
  }
private :
  ITimeMng* m_time_mng ;
  ICollector* m_parent ;
};

#endif

