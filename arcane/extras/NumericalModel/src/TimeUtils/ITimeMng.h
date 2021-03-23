// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ITIMEMNG_H
#define ITIMEMNG_H


#include <arcane/ItemTypes.h>

using namespace Arcane;

/**
 * Interface du service du modu?le de gestion du pas de temps.
 */
 
class ITimeMng
{
public:
  /** Constructeur de la classe */
  ITimeMng() {};
  /** Destructeur de la classe */
  virtual ~ITimeMng() {};
  
public:
  /** 
   *  Initialise 
   */
  virtual void init() = 0;
  virtual void setInitialTime(Real init_time) = 0 ;
  virtual void setFinalTime(Real end_time) = 0 ;  
  virtual void setNewTimeStep(Real deltat) = 0 ;
  virtual void setCurrentTime(Real current_time) = 0 ;
  
  virtual Real getInitialTime() = 0 ;
  virtual Real getFinalTime() = 0 ;
  virtual Real getLastTime() = 0 ;
  virtual Real getCurrentTime() = 0 ;
  virtual Real getNewTimeStep() = 0 ;
  virtual Real getCurrentTimeStep() = 0 ;
  virtual Real getMinTimeStep() = 0 ;
  virtual Real getMaxTimeStep() = 0 ;
  
  virtual bool manageNewTimeStep() = 0 ;
  virtual void startTimeStep() = 0 ;
  virtual void endTimeStep() = 0 ;
  virtual void disableCurrentTimeStep() = 0 ;
  virtual bool isCurrentTimeStepOk() = 0 ;
  
  virtual void stopTimeLoop(bool stop_loop) = 0 ;
  virtual bool timeLoopHasToBeStopped() = 0 ;

  //! Définit le gestionnaire de temps parent
  /// Devrait devenir 'pur virtual' comme les autres méthodes
  virtual void setParent(ITimeMng* parent) { }  
  //! Retourne le gestionnaire de temps parent
  /// Devrait devenir 'pur virtual' comme les autres méthodes
  virtual ITimeMng* getParent() { return NULL ; }
  //! Initialisation à partir du gestionnaire de temps parent
  /// Unification entre initFromParent et init ?
  /// Devrait devenir 'pur virtual' comme les autres méthodes
  virtual void initFromParent() {}  
};


#endif
