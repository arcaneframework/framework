// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef INUMERICALMODELVISITOR_H_
#define INUMERICALMODELVISITOR_H_


/**
  * \author Jean-Marc GRATIEN
  * \version 1.0
  * \brief Interface INumericalModelVisitor de visiteur des INumericalModel
  *
  * Une instance a pour objectif de visiter un INumericalModel concret et de
  * lui appliquer les operations qui lui sont attribuees d'effectuer.
  */

class MultiPhaseFlowNumericalModelService ;
class ReactiveTransportNumericalModelService;
class IDomainModel ;

class INumericalModelVisitor
{
public:
  INumericalModelVisitor() {}
  virtual ~INumericalModelVisitor(){}
  virtual String getName() = 0 ;
  virtual Integer visit(MultiPhaseFlowNumericalModelService* model)
  {
    throw Arcane::FatalErrorException(A_FUNCINFO,"not implemented");
  }
  virtual Integer visit(ReactiveTransportNumericalModelService* model)
    {
      throw Arcane::FatalErrorException(A_FUNCINFO,"not implemented");
    }
  virtual Integer visit(IDomainModel* model)
  {
    throw Arcane::FatalErrorException(A_FUNCINFO,"not implemented");
  }
  virtual Integer visit(INumericalModelVisitor* model)
  {
    throw Arcane::FatalErrorException(A_FUNCINFO,"not implemented");
  }
};

template<typename ModelType>
class ModelVisitorT : public INumericalModelVisitor
{
public:
    virtual ~ModelVisitorT()
    {
    }

    //! to ensure that the class is pure abstract
    virtual String getName() = 0;
    
    virtual Integer visit(ModelType* model)
    {
      throw Arcane::FatalErrorException(A_FUNCINFO,"not implemented");
    }
    virtual Integer visitForStart(ModelType* model)
    {
      throw Arcane::FatalErrorException(A_FUNCINFO,"not implemented");
    }
    virtual Integer visitForFinalize(ModelType* model)
    {
      throw Arcane::FatalErrorException(A_FUNCINFO,"not implemented");
    }
};

#endif /*INUMERICALMODELVISITOR_H_*/
