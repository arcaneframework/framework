// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include "Utils/Utils.h"
#include "TimeUtils/TimeStepMng/VarTimeStepMng.h"
#include <arcane/utils/NotImplementedException.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VarTimeStepMngService::init()
{
    m_type = options()->type() ;
    m_increase_factor = options()->increaseFactor() ;
    m_decrease_factor = options()->decreaseFactor() ;
    m_drvar = 1+options()->varFactor() ;
    m_num_criteria = 0 ;
}

bool 
VarTimeStepMngService::
manageTimeStep(Real* deltat)
{
  VarTimeStepMode mode = computeVarTimeStepMode() ;
  switch(mode)
  {
  case IncreaseTimeStepMode :
    return increaseTimeStep(deltat) ;
  case CutTimeStepMode :
  {
    cutTimeStep(deltat) ;
    switch(m_status)
    {
      case ITimeStepMng::NoError :
      case ITimeStepMng::MinStepReached :
        return true ;
      default :
      return false ;
    }
  }
  case RerunTimeStepMode :
    if(!cutTimeStep(deltat))
      m_status = ITimeStepMng::FatalError ;
    return false ;
  default :
    m_status = ITimeStepMng::NoError ;
    return true ;
  }
}

bool 
VarTimeStepMngService::
cutTimeStep(Real* deltat)
{
    switch(m_type)
    {
        case TypesTimeStepMng::Geometric :
            *deltat *= m_decrease_factor ;
            break ;
        case TypesTimeStepMng::Arithmetic :
            *deltat -= m_decrease_factor ;
            break ;
        default :
            m_status = ITimeStepMng::UndefinedError ;
            return false ;
    }
    if(*deltat < m_deltat_min)
    {
        *deltat = m_deltat_min ;
        m_status = ITimeStepMng::MinStepReached ;
        return false ;
    }
    m_status = ITimeStepMng::NoError ;
    return true ;
} 

bool 
VarTimeStepMngService::
increaseTimeStep(Real* deltat)
{
  switch(m_type)
  {
      case TypesTimeStepMng::Geometric :
          *deltat *= m_increase_factor ;
          break ;
      case TypesTimeStepMng::Arithmetic :
          *deltat += m_increase_factor ;
          break ;
      default :
          m_status = ITimeStepMng::UndefinedError ;
          return false ;
  }
  if(*deltat > m_deltat_max) 
  {
     *deltat=m_deltat_max ;
     m_status = ITimeStepMng::MaxStepReached ;
     info()<<"Max deltat has been reached";
  }
  m_status = ITimeStepMng::NoError ;
  return true ;
}

VarTimeStepMngService::VarTimeStepMode 
VarTimeStepMngService::
computeVarTimeStepMode()
{
  Integer mode = 3 ;
  for(Integer icrit=0;icrit<m_dx.size();icrit++)
  {
    mode = math::min(mode,getVarCriteriaMode(icrit)) ;
  }
  switch(mode)
  {
  case 0:
    return CutTimeStepMode ;
  case 1:
    return NoVarTimeStepMode ;
  case 2:
    return IncreaseTimeStepMode ;
  case -1:
    return RerunTimeStepMode ;
  case 3 :
    return NoVarTimeStepMode ;
  default :
    throw NotImplementedException(A_FUNCINFO);
    return NoVarTimeStepMode ;
  }
}

Integer
VarTimeStepMngService::
getVarCriteriaMode(Integer icrit)
{
  Real dx = m_dx[icrit] ;
  if(dx<m_dx1[icrit]/m_drvar) 
    return 2 ; // increase time step ;
  else if(dx<m_dx1[icrit]*m_drvar)
    return 1 ; // keep time step ;
  else if(dx<m_dx2[icrit])
    return 0 ; // cut time step ;
  else
    return -1 ; // rerun time step ;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_VARTIMESTEPMNG(VarTimeStepMng,VarTimeStepMngService);
