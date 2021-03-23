// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IOP_H
#define IOP_H
class IOp
{
public :
  virtual ~IOp() {}
  virtual void compute() = 0 ;
} ;

class IIntOp
{
public :
  virtual ~IIntOp() {}
  virtual Integer compute() = 0 ;
} ;

class IOpR
{
public :
  virtual ~IOpR() {}
  virtual void compute(Real x) = 0 ;
} ;

template<class T>
class IOpT
{
public :
  virtual ~IOpT() {}
  virtual void compute(T* x) = 0 ;
} ;
template<class T>
class IOpVT
{
public :
  virtual ~IOpVT() {}
  virtual void compute(T x) = 0 ;
} ;

template<class T>
class IOpRefT
{
public :
  virtual ~IOpRefT() {}
  virtual void compute(T& x) = 0 ;
} ;
#endif 


