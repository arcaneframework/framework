// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "NumericalModel/Algorithms/TimeIntegrator/TimeIntegrator.h"

#include "NumericalModel/Models/IIterativeTimeModel.h"

#include "TimeUtils/ITimeMng.h"
#include "TimeUtils/ITimeStepMng.h"

#include <arcane/utils/Math.h>

TimeStepIterator::TimeStepIterator(INumericalModel* model)
{
  IIterativeTimeModel* time_model = dynamic_cast<IIterativeTimeModel*>(model) ;
  if(!time_model)
  {
    m_error = ITimeIntegrator::BadModelTypeError ;
    m_stop = true ;
    return ;
  }
  m_time_mng = time_model->getTimeMng() ;
  m_time_mng->initFromParent() ;
  m_time_step_mng = time_model->getTimeStepMng() ;
  m_init_time = m_time_mng->getInitialTime() ;
  m_final_time = m_time_mng->getFinalTime() ;
  m_current_deltat = m_time_mng->getCurrentTimeStep() ;
  m_last_time = m_init_time ;
  m_current_time = math::min(m_final_time,m_init_time+m_current_deltat) ;
  m_current_deltat = m_current_time-m_init_time ;
  m_error = ITimeIntegrator::NoError ;
  if(m_current_deltat>0)
  {
    //initialisation du temps et du pas de temps courant
    m_time_mng->setNewTimeStep(m_current_deltat) ;
    m_time_mng->endTimeStep() ;
    m_time_mng->setCurrentTime(m_current_time) ;
    m_stop = false ;
  }
  else
    m_stop = true ;
  m_error = ITimeIntegrator::NoError ;
}

TimeStepIterator& TimeStepIterator::operator++()
{
  if(m_time_mng->isCurrentTimeStepOk())
  {
    Real deltat = m_time_mng->getCurrentTimeStep() ;
    bool timeStepOk = m_time_step_mng->manageTimeStep(&deltat) ;
    if(timeStepOk)
    {
      m_time_mng->setNewTimeStep(deltat) ;
    }
    else
      m_time_mng->disableCurrentTimeStep() ;
  }
  const bool isNewTimeStepOk = m_time_mng->manageNewTimeStep() ;

  if (not isNewTimeStepOk)
  {
    m_stop = true ;
    m_error = ITimeIntegrator::TimeStepError ;
    return *this ;
  }
  if(m_time_mng->isCurrentTimeStepOk())
    {
      m_time_mng->endTimeStep();
      m_current_deltat = m_time_mng->getCurrentTimeStep();
      if(m_current_time >= m_time_mng->getFinalTime())
        {
          m_stop = true;
        }
      else
        {
          if (m_current_deltat==0.)
            {
              m_error = ITimeIntegrator::TimeStepError;
              m_stop = true;
              return *this;
            }
          m_last_time = m_current_time;
          m_current_time += m_current_deltat;
          m_time_mng->setCurrentTime(m_current_time);
        }
    }
  else
  {
    // Le pas de temps doit etre rejoue avec un nouveau pas de temps
    m_time_mng->endTimeStep() ;
    m_current_deltat = m_time_mng->getCurrentTimeStep() ;
    m_current_time = m_last_time + m_current_deltat;
    m_time_mng->setCurrentTime(m_current_time) ;
  }
  return *this ;
}

TimeIntegrator::TimeIntegrator()
: IntegratorT<TimeStepIterator>()
{}

