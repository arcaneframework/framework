// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ITIMESTEPMNG_H
#define ITIMESTEPMNG_H


#include <arcane/ItemTypes.h>

using namespace Arcane;

/**
 * Interface du service du modu?le de gestion du pas de temps.
 */
 
class ITimeStepMng
{
public:
  typedef enum {
    NoError,
    FatalError,
    MaxStepReached,
    MinStepReached,
    UndefinedError
  } eStatusType ;
  
  /** Constructeur de la classe */
  ITimeStepMng() {};
  /** Destructeur de la classe */
  virtual ~ITimeStepMng() {};
  
public:
  /** 
   *  Initialise 
   */
  virtual void init() = 0;
  virtual void setMaxTimeStep(Real max_deltat) = 0 ;
  virtual void setMinTimeStep(Real min_deltat) = 0 ;
  
  /**
   *  gere la variation du pas de temps
   */
  virtual bool manageTimeStep(Real* deltat) = 0 ;
  /**
   * coupe le pas de temps
   */
  virtual bool cutTimeStep(Real* deltat) = 0 ;
  virtual bool increaseTimeStep(Real* deltat) = 0 ;
  
  virtual eStatusType getStatus() = 0 ;
};


#endif
