// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ICOLLECTOR_H
#define ICOLLECTOR_H
class IOp ;
class ICollector
{
public :
  //!Action type during one computation sequence
  typedef enum
  {
     Start,
     StartCompute,
     Compute,
     FinalizeCompute,
     Finalize
  } eActionType ;

  //!desctrutor
  virtual ~ICollector() {}
  virtual String getName() const = 0 ;

  //!record operator and manage operator life
  virtual void addOperator(eActionType type,IOp* op) = 0 ;

  //! compute operators of one Action type
  virtual void compute(eActionType type) = 0 ;

  //! compute operator of Start type
  virtual void start() = 0 ;

  //! compute all operators of StartCompute type
  virtual void startcompute() = 0 ;

  //! compute all operators of Compute type
  virtual void compute() = 0 ;

  //! compute all operators of FinalizeCompute type
  virtual void finalizecompute() = 0 ;

  //! compute all operators of StartCompute, Compute and FinalizeCompute type
  virtual void computeAll() 
  {
     startcompute() ;
     compute() ;
     finalizecompute() ;
  }

  //! compute all operators of FinalizeCompute type
  virtual void finalize() = 0 ;
};
#endif
