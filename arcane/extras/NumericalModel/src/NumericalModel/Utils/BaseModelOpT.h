// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef BASEMODELOPT_H
#define BASEMODELOPT_H

class ICollector ;
template<class Model>
class BaseModelOpT : public Model::Visitor
{
public:
  BaseModelOpT(ICollector* collector)
  : m_parent(collector)
  {
  }
  virtual ~BaseModelOpT()
  {
  }
  String getName() { return "BaseModelOpT" ; }

  virtual Integer visit(Model* model)
  {
    if(m_parent)
      m_parent->computeAll() ;
    return model->getError() ;
  }
  virtual Integer visitForStart(Model* model)
  {
    model->setVars(m_parent) ;
    if(m_parent)
      m_parent->start() ;
    return model->getError() ;
  }
  virtual Integer visitForFinalize(Model* model)
  {
    if(m_parent)
      m_parent->finalize() ;
    return model->getError() ;
  }
private :
  ICollector* m_parent ;
};

#endif

