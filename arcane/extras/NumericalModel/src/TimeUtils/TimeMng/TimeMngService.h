// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef TIMEMNGSERVICE_H
#define TIMEMNGSERVICE_H

#include "TimeUtils/ITimeMng.h"
#include "TimeMng_axl.h"

using namespace Arcane ;

class TimeMngService :
    public ArcaneTimeMngObject
{
public:
  /** Constructeur de la classe */
  TimeMngService(const ServiceBuildInfo & sbi) 
    : ArcaneTimeMngObject(sbi) 
    , m_parent(NULL)
    , m_init_time(0)
    , m_min_time_step(0)
    , m_max_time_step(0)
    , m_current_time_step(0)
    , m_new_time_step(0)
    , m_restore_time_step(0)
    , m_step_is_Ok(true)
    , m_time_step_has_to_be_restored(false)
    , m_time_loop_has_to_be_stopped(false)
    , m_initialized(false) 
    {}
  
  /** Destructeur de la classe */
  virtual ~TimeMngService() {};
  
public:
  /** 
   *  Initialise 
   */
  void init() ;
   
  void setInitialTime(Real init_time)
  {
    m_init_time = init_time ;
  }
  
  void setFinalTime(Real end_time)
  {
    m_global_final_time = end_time ;
  }
  
  void setDeltatMax(Real max_deltat) 
  {
    m_max_time_step = max_deltat ;
  }
  
  void setDeltatMin(Real min_deltat) 
  {
    m_min_time_step = min_deltat ;
  }
  
  //! Propose un nouveau pas de temps
  void setNewTimeStep(Real deltat) ;
  
  Real getNewTimeStep() ;
  Real getCurrentTimeStep() ;
  Real getMinTimeStep() { return m_min_time_step ; }
  Real getMaxTimeStep() { return m_max_time_step ; }
  
  Real getInitialTime() { return m_init_time ; }
  Real getFinalTime() 
  { 
   if(m_parent)
     return m_local_final_time ;
   else
    return m_global_final_time() ; 
  }
  Real getCurrentTime() ;
  Real getOldCurrentTime() ;
  
  //Set Current time (when m_parent is not global Arcane TimeLoop Mng)
  void setCurrentTime(Real current_time) ;

  Real getLastTime() { return m_global_old_time() ; }


  void startTimeStep() ;
  void endTimeStep() ;
  bool manageNewTimeStep() ;
  void disableCurrentTimeStep() { m_step_is_Ok = false ; }
  bool isCurrentTimeStepOk() { return m_step_is_Ok ; }
  bool timeLoopHasToBeStopped() { return m_time_loop_has_to_be_stopped ; }
  

  void stopTimeLoop(bool stop_loop) ;

  void setParent(ITimeMng* parent) 
  { m_parent = parent ;}
  ITimeMng* getParent()
  { return m_parent ; }

  void initFromParent() ;

private:
  ITimeMng* m_parent ;

  Real m_init_time ;
  Real m_min_time_step ;
  Real m_max_time_step ;
  Real m_current_time_step ;
  Real m_old_current_time ;
  Real m_current_time ;
  Real m_new_time_step ;
  Real m_restore_time_step ;
  bool m_step_is_Ok ;
  bool m_time_step_has_to_be_restored ;
  bool m_time_loop_has_to_be_stopped ;
  bool m_initialized ;
  
  Real m_local_final_time ;

  typedef enum {
    MinTimeStep,
    MaxTimeStep,
    InitTimeStep
  } eTimeStepProperty ;
};

//END_NAME_SPACE_PROJECT

#endif
