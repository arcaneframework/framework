// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef INUMERICALMODEL_H
#define INUMERICALMODEL_H
/**
 * \author Jean-Marc GRATIEN
 * \version 1.0
 * \brief Interface du service du modèle numerique de résolution des systèmes
 *  d'équations de type EDP.
 *
 */

#include <arcane/utils/FatalErrorException.h>
#include <arcane/utils/TraceInfo.h>

using namespace Arcane;

template<class Model>
struct ModelTraits
{
  typedef void(Model::*func_type)() ;
  typedef void(Model::*funcReal_type)(Real) ;
  typedef Integer (Model::*ifunc_type)() ;
  typedef Integer (Model::*ifuncReal_type)(Real) ;
  template<class T>
  struct FuncType
  {
     typedef void(Model::*funcT_type)(T*) ;
     typedef void(Model::*funcVT_type)(T) ;
     typedef void(Model::*funcRefT_type)(T&) ;
     typedef Integer(Model::*ifuncT_type)(T*) ;
     typedef Integer(Model::*ifuncVT_type)(T) ;
     typedef Integer(Model::*ifuncRefT_type)(T&) ;
  } ;
};

class ICollector ;
class INumericalDomain ;
class IPostMng ;
class INumericalModelVisitor ;

class INumericalModel
{
public:
  /** Constructeur de la classe */
  INumericalModel() {}
  /** Destructeur de la classe */
  virtual ~INumericalModel() {}

  //! return model parent
  virtual INumericalModel* getParentModel() const { return NULL ; }
  virtual void setParentModel(INumericalModel* parent) {}

  virtual String getName() const = 0 ;
public:
  //!Initialise
  virtual void init() = 0;
  virtual void startInit() {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  virtual void prepareInit() {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }

  //! Reprise
  virtual void continueInit() = 0 ;

  //! start computation
  virtual void start() = 0;
  virtual void start(Integer sequence) {
    start() ;
  }

  //base compute sequence without start and finalize
  virtual Integer baseCompute() = 0 ;

  //compute sequence with start and finalize
  virtual Integer compute()
  {
    start() ;
    Integer code = baseCompute() ;
    finalize() ;
    return code ;
  }

  //! compute one sequence of operators with start and finalize
  virtual Integer compute(Integer sequence) {
    start(sequence) ;
    Integer code = baseCompute(sequence) ;
    finalize(sequence) ;
    return code ;
  }
  virtual Integer baseCompute(Integer sequence) {
    return baseCompute() ;
  }

  virtual void finalize() = 0;

  virtual void prepare(ICollector* collector, Integer sequence) {}
  virtual void finalize(Integer sequence) {
    finalize() ;
  }

  virtual void update() = 0 ;
  virtual INumericalDomain* getINumericalDomain() = 0 ;

  virtual IPostMng* getPostMng() = 0 ;

  virtual Integer accept(INumericalModelVisitor* visitor) = 0 ;

  class ISequenceObserver
  {
  public :
    virtual ~ISequenceObserver() {} ;
    virtual void update() = 0 ;
  };

  //virtual void notifyNewSequence(Integer sequence) = 0 ;
  //virtual void addObs(ISequenceObserver* obs,Integer sequence) = 0 ;
};
 //END_NAME_SPACE_PROJECT


#endif
