// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef VARTIMESTEPMNG_H
#define VARTIMESTEPMNG_H

#include "TimeUtils/ITimeStepMng.h"
#include "TimeUtils/TimeStepMng/TypesTimeStepMng.h"
#include "VarTimeStepMng_axl.h"

using namespace Arcane ;

/**
 * Interface du service du modu?le de resolution non lineaire.
 */

class VarTimeStepMngService :
    public ArcaneVarTimeStepMngObject
{
public:
  /** Constructeur de la classe */
  VarTimeStepMngService(const ServiceBuildInfo & sbi) 
  : ArcaneVarTimeStepMngObject(sbi)
  , m_status(ITimeStepMng::NoError)
  {}
  
  /** Destructeur de la classe */
  virtual ~VarTimeStepMngService() {};
  
public:
  /** 
   *  Initialise 
   */
  void init() ;
  
  void setMaxTimeStep(Real max_deltat) 
  {
    m_deltat_max = max_deltat ;
  }
  
  void setMinTimeStep(Real min_deltat) 
  {
    m_deltat_min = min_deltat ;
  }
  
  /**
   *  gere la variation du pas de temps
   */
  virtual bool manageTimeStep(Real* deldat) ;
  /**
   * coupe le pas de temps
   */
  virtual bool cutTimeStep(Real* deldat) ;
  virtual bool increaseTimeStep(Real* deldat) ;

  ITimeStepMng::eStatusType getStatus() {
	  return m_status ;
  }
 private :
    TypesTimeStepMng::eTimeStepMngType m_type ;
    Real m_increase_factor ;
    Real m_decrease_factor ;
    Real m_drvar ;
    
    Real m_deltat_max ;
    Real m_deltat_min ;
    
    typedef enum {
      CutTimeStepMode,
      IncreaseTimeStepMode,
      NoVarTimeStepMode,
      RerunTimeStepMode
    } VarTimeStepMode ;
    
    VarTimeStepMode computeVarTimeStepMode() ;
    Integer getVarCriteriaMode(Integer icrit) ;
    
    
    ITimeStepMng::eStatusType m_status ;
};

//END_NAME_SPACE_PROJECT

#endif
