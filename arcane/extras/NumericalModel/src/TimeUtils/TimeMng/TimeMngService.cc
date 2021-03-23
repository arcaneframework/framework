// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include "Utils/Utils.h"

#include "PropertyUtils/PropertyHolder.h"
#include "Appli/IAppServiceMng.h"

#include "TimeUtils/TimeMng/TimeMngService.h"

#include <arcane/ITimeLoopMng.h>
#include <arcane/ISubDomain.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
TimeMngService::
init()
{
  if(m_initialized) return ;
    
  m_init_time = options()->initTime() ;
  m_global_final_time = options()->endTime() ;
  m_new_time_step = options()->initTimeStep() ;
  m_min_time_step = options()->minTimeStep() ;
  m_max_time_step = options()->maxTimeStep() ;
    
  // Préparation minimale
  // Attention donnée peu compatible avec une reprise...
  m_step_is_Ok = true;
  m_time_step_has_to_be_restored = false;
  m_global_old_time = m_init_time;
  m_global_time = m_init_time;
  
  m_initialized = true ;
}

Real 
TimeMngService::
getCurrentTime()
{
  if(m_parent)
    return m_current_time ;
  else
    return m_global_time() ;
}
Real 
TimeMngService::
getOldCurrentTime()
{
  if(m_parent)
    return m_old_current_time ;
  else
    return m_global_old_time() ;
}
void  
TimeMngService::
setCurrentTime(Real current_time)
{
  if(m_parent)
  {
    m_old_current_time = m_current_time ;
    m_current_time = current_time ;
  }
}
void 
TimeMngService::
setNewTimeStep(Real deltat)
{
   info()<<"TimeMngService::setNewTimeStep "<<deltat;
   m_new_time_step = math::min(m_new_time_step,deltat) ;
} 

Real 
TimeMngService::
getNewTimeStep()
{
  return m_new_time_step ;
}

Real 
TimeMngService::
getCurrentTimeStep()
{
  return m_current_time_step ;
}

bool 
TimeMngService::
manageNewTimeStep()
{
  // - si le pas de temps n'est pas annulé, 
  // - si le pas de temps était artificiellement réduit pour répondre
  //   à une contrainte non physique (évènement)
  // - si le pas de temps proposé par les modules physiques n'a pas 
  //   été réduit (en prévision d'une difficulté numérique)
  // => on le restore 

  debug() << "TimeMngService::manageNewTimeStep " << m_new_time_step;

  if (m_step_is_Ok and
      m_new_time_step >= m_global_deltat() and 
      m_time_step_has_to_be_restored) 
    {
      m_new_time_step = math::max(m_new_time_step,m_restore_time_step);
    }
  m_time_step_has_to_be_restored = false;

  // On verifie que le pas de temps demande
  // par la physique n'est pas sous le dtmin
  if(m_new_time_step<m_min_time_step)
    {
      error() << "Min deltat has been reached : " << m_new_time_step << " < " << m_min_time_step;
      return false;
    }
    
  // Si le pas de temps demandé par la physique
  // est supérieur au dtmax, on utilise dtmax
  if(m_new_time_step>m_max_time_step)
    {
      warning() << "Max deltat has been reached : " << m_new_time_step << " > " << m_max_time_step;
      m_new_time_step = m_max_time_step ;
    }
    
  // Temps courant (adapté si le temps a été annulé)
  const Real current_time = (m_step_is_Ok)?getCurrentTime():getOldCurrentTime();

  // Le pas de temps maximal est défini par le temps final
  Real next_time_event = m_global_final_time();

  // On reduit le pas de temps eventuellement 
  // pour passer par les temps exigés dans l'EventMng
  debug() << "nextTimeEvent(" << current_time << ") : " << next_time_event;
  ARCANE_ASSERT((current_time<=next_time_event),("Inconsistent result"));

  if (current_time+m_new_time_step>=next_time_event)
    {
      m_restore_time_step = m_new_time_step ;
      m_new_time_step = next_time_event-current_time ;
      m_time_step_has_to_be_restored = true ;
    }
  else if (current_time+m_new_time_step>next_time_event-m_min_time_step)
    {
      // Pour eviter d'etre sous le pas de temps min au prochain
      // pas de temps on reduit le pas de temps par 2
      m_restore_time_step = m_new_time_step ;
      m_new_time_step = (next_time_event-current_time)/2. ;
      m_time_step_has_to_be_restored = true ;
    }

  return true ;
}

void 
TimeMngService::
startTimeStep()
{
  // initialise avec la garantie d'un majorant du pas de temps autorisé
  // (pour faire des math::min dessus ensuite)
  m_new_time_step = m_max_time_step + 1 ;
}

void
TimeMngService::
endTimeStep()
{
  m_step_is_Ok = true ;
  m_current_time_step = m_new_time_step ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
TimeMngService::
initFromParent()
{
  if(m_parent)
  {
     m_init_time = m_parent->getLastTime() ;
     m_local_final_time  = m_parent->getCurrentTime() ;
     m_current_time_step = options()->initTimeStep() ;
     m_min_time_step     = options()->minTimeStep() ;
     m_max_time_step     = options()->maxTimeStep() ;
     m_new_time_step = m_current_time_step ;
     m_old_current_time = m_init_time ;
     m_current_time = m_init_time ;
  }
}

void
TimeMngService:: 
stopTimeLoop(bool stop_loop) {
  m_time_loop_has_to_be_stopped = true ;
  if(stop_loop)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

ARCANE_REGISTER_SERVICE_TIMEMNG(TimeMng,TimeMngService);
