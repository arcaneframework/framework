// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef OPT_H
#define OPT_H

#include "NumericalModel/Utils/IOp.h"
#include "NumericalModel/Models/INumericalModel.h"

using namespace Arcane;

template<class Model>
class OpT : public IOp
{
public :
  typedef typename ModelTraits<Model>::func_type func_type ;
  OpT(Model* model,func_type func)
  : m_model(model)
  , m_func(func)
  {}
  virtual ~OpT() {}
  virtual void compute() 
  {
     (m_model->*m_func)() ;
  }
private :
  Model* m_model ;
  func_type m_func ;
} ;

template<class Model>
class IntOpT : public IIntOp
{
public :
  typedef typename ModelTraits<Model>::ifunc_type func_type ;
  IntOpT(Model* model,func_type func)
  : m_model(model)
  , m_func(func)
  {}
  virtual ~IntOpT() {}
  virtual Integer compute() 
  {
     return (m_model->*m_func)() ;
  }
private :
  Model* m_model ;
  func_type m_func ;
} ;

template<class Model>
class OpRT : public IOpR
{
public :
  OpRT(Model* model,typename ModelTraits<Model>::funcR_type func)
  : m_model(model)
  , m_func(func)
  {}
  virtual ~OpRT() {}
  virtual void compute(Real x) 
  {
     (m_model->*m_func)(x) ;
  }
private :
  Model* m_model ;
  typename ModelTraits<Model>::funcR_type m_func ;
} ;

template<class Model,class T>
class OpTT : public IOpT<T>
{
public :
  typedef typename ModelTraits<Model>::template FuncType<T>::funcT_type func_type ;
  OpTT(Model* model,func_type func)
  : m_model(model)
  , m_func(func)
  {}
  virtual ~OpTT() {}
  virtual void compute(T* x) 
  {
     (m_model->*m_func)(x) ;
  }
private :
  Model* m_model ;
  func_type m_func ;
} ;
template<class Model,class T>
class OpTVT 
: public IOpVT<T>
, public IOp
{
public :
  typedef typename ModelTraits<Model>::template FuncType<T>::funcVT_type func_type ;
  OpTVT(Model* model,func_type func,T x)
  : m_model(model)
  , m_func(func)
  , m_x(x)
  {}
  virtual ~OpTVT() {}
  virtual void compute(T x) 
  {
     (m_model->*m_func)(x) ;
  }
  virtual void compute() 
  {
     (m_model->*m_func)(m_x) ;
  }
private :
  Model* m_model ;
  func_type m_func ;
  T m_x ;
} ;

template<class Model,class T>
class OpTRefT 
: public IOpRefT<T>
, public IOp
{
public :
  typedef typename ModelTraits<Model>::template FuncType<T>::funcRefT_type func_type ;
  OpTRefT(Model* model,func_type func,T x=T())
  : m_model(model)
  , m_func(func)
  , m_x(x)
  {}
  virtual ~OpTRefT() {}
  virtual void compute(T& x) 
  {
     (m_model->*m_func)(x) ;
  }
  virtual void compute() 
  {
     (m_model->*m_func)(m_x) ;
  }
private :
  Model* m_model ;
  func_type m_func ;
  T m_x ;
} ;
#endif

