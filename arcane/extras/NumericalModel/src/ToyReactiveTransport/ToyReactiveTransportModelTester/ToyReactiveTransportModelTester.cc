// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Utils/Utils.h"

#include <boost/shared_ptr.hpp>

#include "Tests/ServiceTesters/IServiceTester.h"
#include "Tests/IServiceValidator.h"

#include "Utils/ItemGroupMap.h"
#include "Utils/ItemGroupBuilder.h"

#include "Numerics/DiscreteOperator/IDivKGradDiscreteOperator.h"

#include "NumericalModel/Utils/ICollector.h"
#include "NumericalModel/Utils/IOp.h"
#include "NumericalModel/Models/INumericalModel.h"
#include "NumericalModel/Operators/INumericalModelVisitor.h"
#include "NumericalModel/SubDomainModel/INumericalDomain.h"

#include "NumericalModel/Utils/BaseCollector.h"
#include "NumericalModel/Utils/OpT.h"
#include "NumericalModel/SubDomainModel/NumericalDomain/NumericalDomainImpl.h"
#include "NumericalModel/SubDomainModel/SubDomainModelProperty.h"
#include "NumericalModel/SubDomainModel/CollectorT.h"
#include "NumericalModel/SubDomainModel/SDMBoundaryCondition.h"
#include "NumericalModel/SubDomainModel/SDMBoundaryConditionMng.h"
#include "NumericalModel/Models/ISubDomainModel.h"

#include "Mesh/Geometry/IGeometryMng.h"
#include "TimeUtils/ITimeMng.h"
#include "TimeUtils/ITimeStepMng.h"


#include "ToyReactiveTransport/ToyReactiveTransportModelTester/ToyReactiveTransportModelTester.h"

//#include "Appli/IAppServiceMng.h"

#include "Mesh/Geometry/IGeometryMng.h"

#include "NumericalModel/Models/IIterativeTimeModel.h"
//#include "NumericalModel/Algorithms/Integrator/IntegratorT.h"
//#include "NumericalModel/Algorithms/TimeIntegrator/TimeIntegrator.h"
#include "NumericalModel/FluxModel/FluxModel.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/

void ToyReactiveTransportModelTester::init() 
{
  info() << "init";
  m_output_level = options()->outputLevel() ;
  m_validator = NULL ;

   // lecture des sous domaines et schemas associes
  m_model = options()->model();
  m_model->init() ;
  
  //Domain management
  initModelDomains() ;
  
  //Time step management
  //IAppServiceMng* app_service_mng = IAppServiceMng::instance(subDomain()->serviceMng()) ;
  //m_time_mng = app_service_mng->find<ITimeMng>(true) ;
  m_time_mng = options()->timeMng() ;
  m_time_mng->init() ;
  m_time_step_mng = options()->timeStepMng() ;
  m_time_step_mng->init() ;
  {
    //
    // GLOBAL DOMAIN MODEL
    //
    IIterativeTimeModel* model = dynamic_cast<IIterativeTimeModel*>(m_model) ;
    if(model) 
      model->setTimeMng(m_time_mng);

    ISubDomainModel* sd_model = dynamic_cast<ISubDomainModel*>(m_model) ;
    if(sd_model)
    {
       ISubDomainModel::FaceBoundaryConditionMng* bc_mng = sd_model->getFaceBoundaryConditionMng() ;
       
       setBoundaryCondition(sd_model,m_bc_group) ;
       
       m_collector = new ISubDomainModel::Collector(m_time_mng,bc_mng) ;
       m_model->prepare(m_collector,DtSeq) ;
    }
  }
  //Initial state computation
  computeInitialState() ;
   
   //////////////////////////////
   //    START COMPUTATION     //
   //////////////////////////////
  m_error = 0 ;
  m_model->start() ;
  //Fin de pas de temps fictif
  m_time_mng->endTimeStep() ; 
  m_global_deltat = m_time_mng->getCurrentTimeStep() ;
 }

void
ToyReactiveTransportModelTester::
initModelDomains()
{
  // lecture des groupes de mailles
  readGroup();
  {
    ISubDomainModel::NumericalDomain* domain = 
      dynamic_cast<ISubDomainModel::NumericalDomain*> (m_model->getINumericalDomain()) ;
    if(domain)
    {
      domain->setName("GlobalDomain") ;
      domain->setInternalItems(allCells(), m_internal_face) ;
      domain->setBoundary(m_boundary_face) ;
      for(Integer i=0;i<m_bc_group.size();i++)
        if(m_bc_group[i].size()>0)
          domain->addFaceBoundary(m_bc_group[i],i) ;
      //domain->setInterfaceItems(m_overlap_group, m_overlap_inner_group) ;

//      if(m_output_level>1)
//        domain->printInfo() ;
    }
    else
      fatal()<<"Numerical model should be a domain model";
  }
}

/*---------------------------------------------------------------------------*/
// Lecture des groupes de mailles : reservoir et window
// puis construction du groupe de maille overlap et de tous les groupes de
// faces et de nodes  necessaires aux calculs
/*---------------------------------------------------------------------------*/
void ToyReactiveTransportModelTester::readGroup()
{
  
  // Construction d'un groupe des faces internes
  ItemGroupBuilder<Face> iface_builderGlobal(allFaces().mesh(),"GLOBALINTERNALFACEGROUP");
  ItemGroupBuilder<Face> bface_builderGlobal(allFaces().mesh(),"GLOBALBOUNDARYFACEGROUP");

  // 
  // maillage global : 
  // 
  // Ensemble des cellules internes (Integer est une valeur fictive)
  ItemGroupMapT<Cell,Integer> isInnerCellGlobal(allCells());

  ENUMERATE_FACE(iface, allFaces())
  {
    const Face & face = *iface;
    if (isInnerCellGlobal.hasKey(face.frontCell()) and isInnerCellGlobal.hasKey(face.backCell()))
    {
      iface_builderGlobal.add(face);
    }
    else
    {
      ARCANE_ASSERT((isInnerCellGlobal.hasKey(face.frontCell()) xor 
                    isInnerCellGlobal.hasKey(face.backCell())),
                    ("Group edge computation") );
      bface_builderGlobal.add(face);
    }
  }
  m_internal_face = iface_builderGlobal.buildGroup();
  m_boundary_face = bface_builderGlobal.buildGroup();

  //
  // CREATE USERDEFINED BOUNDARIES
  //
   ItemGroupMapT<Face,Integer> isBoundaryFace(m_boundary_face);
  // boucle sur les conditions aux limites
  Integer nb_boundary_condition = options()->boundaryCondition.size();
  m_bc_group.resize(nb_boundary_condition+1) ;
  for (int i = 0; i < nb_boundary_condition; ++i)
  {
    FaceGroup face_group = options()->boundaryCondition[i]->surface();
    m_bc_group[i] = face_group ;
    ENUMERATE_FACE(iface,face_group)
    {
      const Face& face = *iface ;
      isBoundaryFace[face] = 1 ;
    }
  }
  
  String name("DefaultBOUNDARY") ;
  ItemGroupBuilder<Face> builder(m_boundary_face.mesh(),name) ;
  ENUMERATE_FACE(iface,m_boundary_face)
  {
     const Face& face = *iface ;
     if(isBoundaryFace[face]==0)
        builder.add(face) ;
  }
  m_bc_group[nb_boundary_condition] = builder.buildGroup() ;
}

void ToyReactiveTransportModelTester::
setBoundaryCondition(ISubDomainModel* sd_model,
                     Array<FaceGroup>& bc_group )
{
  ISubDomainModel::FaceBoundaryConditionMng* bc_mng = sd_model->getFaceBoundaryConditionMng() ;
  // boucle sur les conditions aux limites
  Integer nb_boundary_condition = options()->boundaryCondition.size();
  for (int i = 0; i < nb_boundary_condition; ++i)
  {
    FaceGroup face_group =  bc_group[i] ;
    if(face_group.size()>0)
    {
      Real valeurBC = options()->boundaryCondition[i]->value();
      SubDomainModelProperty::eBoundaryConditionType typeBC = options()->boundaryCondition[i]->type();
    
      // CREATE BOUNDARY CONDITION BC0 for boundary 0 and OPERATORS to init them
      ISubDomainModel::FaceBoundaryCondition* bc = new ISubDomainModel::FaceBoundaryCondition(i,typeBC,i) ;
      bc->setValue(valeurBC) ; 
      bc->activate(true) ;
      //Record to bc manager
      bc_mng->addNew(bc,new ISubDomainModel::FaceBCOp(sd_model,&ISubDomainModel::initBoundaryCondition,bc)) ;
    }
  }
  {
    Integer i = nb_boundary_condition ;
    FaceGroup face_group =  bc_group[i] ;
    if(face_group.size()>0)
    {
      SubDomainModelProperty::eBoundaryConditionType typeBC = SubDomainModelProperty::NullFlux ;
    
      // CREATE BOUNDARY CONDITION BC0 for boundary 0 and OPERATORS to init them
      ISubDomainModel::FaceBoundaryCondition* bc = new ISubDomainModel::FaceBoundaryCondition(i,typeBC,i) ;
      bc->setValue(0) ;
      //Record to bc manager
      bc_mng->addNew(bc,new ISubDomainModel::FaceBCOp(sd_model,&ISubDomainModel::initBoundaryCondition,bc)) ;
    }
  }
}
/*---------------------------------------------------------------------------*/

void ToyReactiveTransportModelTester::
computeInitialState()
{
  _copy(allCells(),m_u_concentration_tn,m_u_concentration) ;
  _copy(allCells(),m_v_concentration_tn,m_v_concentration) ;
}

int ToyReactiveTransportModelTester::test()
{
  info() << "compute";
  m_time_mng->startTimeStep() ;
  if (m_output_level>0) 
  {
    info();
    info()<<"|-----------------------------------------------------|";
    info()<<"| T(N)            | T(N+1)          | DT              |";
    info()<<"|"<<FORMAT(10,5)<<m_time_mng->getLastTime()<<"s|"
               <<FORMAT(10,5)<<m_global_time()<<"s|"
               <<FORMAT(10,5)<<m_global_deltat()<<"s|";
    info()<<"|-----------------------------------------------------|";
    info();
  }
  m_u_concentration.synchronize();
  m_v_concentration.synchronize();
  m_permeability.synchronize();
  computeDomain() ;
  computeDeltaT() ;
  return 1 ;
}


void 
ToyReactiveTransportModelTester::
computeDeltaT()
{
  const Real last_time = m_time_mng->getLastTime();
  // Mise ? jour du pas de temps
  if(m_time_mng->isCurrentTimeStepOk())
  {
    if (m_output_level>0) 
    {
      info()<<"|-----------------------------------------------------|";
      info()<<"| T(N)            | T(N+1)          | DT              |";
      info()<<"|"<<FORMAT(10,5)<<last_time<<"s|"
            <<FORMAT(10,5)<<m_global_time()<<"s|"
            <<FORMAT(10,5)<<m_global_deltat()<<"s|";
      info()<<"|-----------------------------------------------------|";
    }
    Real deltat = m_time_mng->getCurrentTimeStep() ;
    bool timeStepOk = m_time_step_mng->manageTimeStep(&deltat) ;
    m_time_mng->setNewTimeStep(deltat) ;
    if(!timeStepOk)
      m_time_mng->disableCurrentTimeStep() ;
  }
}

void ToyReactiveTransportModelTester::
computeDomain()
{
  if(m_output_level>0)
  {
    info() << " Resolution du domain global  ";
  }
  //
  // resolution d'un pas de temps
  //
  m_model->compute(DtSeq);
}
bool ToyReactiveTransportModelTester::goOn()
{
  const Real last_time = m_time_mng->getLastTime();
  if(m_time_mng->isCurrentTimeStepOk())
    {
      m_time_mng->endTimeStep() ;
      m_global_deltat = m_time_mng->getCurrentTimeStep() ;
      if(m_global_time() >= m_time_mng->getFinalTime())
        {
          if (m_global_deltat()!=0.)
            error() << "Final time step inconsistent"; // non fatal mais bizarre
          if(m_validator)
          {
            Integer error = m_validator->validate() ;
            if(error)
              info()<<"VALIDATION FAILED";
            else
              info()<<"STOP PERFECT";
          }
          else
            info()<<"STOP PERFECT";
          return false ;
        }
    }
  else
    {
      // Le pas de temps doit etre rejoue avec un nouveau pas de temps
      m_global_time = last_time ;
      m_time_mng->endTimeStep() ;
      m_global_deltat = m_time_mng->getCurrentTimeStep() ;
      info()<<"|------------------------------|";
      info()<<"|Time Step has to be run again |";
      info()<<"|Old Time "<<FORMAT(10,5)<<last_time<<"|";
      info()<<"|------------------------------|";
    }
  return true ;
}

ARCANE_REGISTER_SERVICE_TOYREACTIVETRANSPORTMODELTESTER(ToyReactiveTransportModelTester,ToyReactiveTransportModelTester);
