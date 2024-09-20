// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleSimpleHydro.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Module Hydrodynamique simple.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

// En mode check, ajoute les traces. Ne le fait pas en mode release
// car cela peut géner la détection des indices de boucle du compilateur et
// empêcher la vectorisation

#ifdef ARCANE_CHECK
#define ARCANE_TRACE_ENUMERATOR
#endif

#include "arcane/utils/List.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/GoBackwardException.h"
#include "arcane/utils/Profiling.h"

#include "arcane/core/BasicModule.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/MathUtils.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Concurrency.h"
#include "arcane/core/VariableView.h"

#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshMng.h"

#include "arcane/tests/TypesSimpleHydro.h"
#include "arcane/tests/SimpleHydro_axl.h"
#include "arcane/tests/TestTraceMessageListener.h"

#include "arcane/hyoda/Hyoda.h"
#include "arcane/core/ItemLoop.h"
#include "arcane/core/ITimeHistoryMng.h"

#include "arcane/core/MeshUtils.h"

// Force la vectorisation avec GCC.
#ifdef __GNUC__
#  pragma GCC optimize ("-ftree-vectorize")
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SimpleHydro
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleHydroModuleBase::getDeltatInit() { return m_options->deltatInit(); }
TypesSimpleHydro::eViscosity SimpleHydroModuleBase::getViscosity(){ return m_options->viscosity(); }
Real SimpleHydroModuleBase::getViscosityLinearCoef(){ return m_options->viscosityLinearCoef(); }
Real SimpleHydroModuleBase::getViscosityQuadraticCoef(){ return m_options->viscosityQuadraticCoef(); }
ConstArrayView<IBoundaryCondition*> SimpleHydroModuleBase::getBoundaryConditions(){ return m_options->getBoundaryCondition(); }
Real SimpleHydroModuleBase::getCfl(){ return m_options->cfl(); }
Real SimpleHydroModuleBase::getVariationSup(){ return m_options->variationSup(); }
Real SimpleHydroModuleBase::getVariationInf(){ return m_options->variationInf(); }
Real SimpleHydroModuleBase::getDensityGlobalRatio(){ return m_options->densityGlobalRatio(); }
Real SimpleHydroModuleBase::getDeltatMax(){ return m_options->deltatMax(); }
Real SimpleHydroModuleBase::getDeltatMin(){ return m_options->deltatMin(); }
Real SimpleHydroModuleBase::getFinalTime(){ return m_options->finalTime(); }
Integer SimpleHydroModuleBase::getBackwardIteration(){ return m_options->backwardIteration(); }
bool SimpleHydroModuleBase::isCheckNumericalResult() { return m_options->checkNumericalResult(); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module hydrodynamique simplifié.
 *
 * Ce module implémente une hydrodynamique simple tri-dimensionnel,
 * parallèle, avec une pseudo-viscosité aux mailles.
 */
class ModuleSimpleHydro
: public ArcaneSimpleHydroObject
{
 public:

  class SecondaryVariables
  {
   public:
    SecondaryVariables(IModule* module)
    : m_faces_density(VariableBuildInfo(module,"FacesDensity")),
      m_unused_variable(VariableBuildInfo(module,"UnusedVariable"))
      {
      }
   public:
    VariableFaceReal m_faces_density;
    VariableCellReal m_unused_variable;    
  };

 public:

  //! Constructeur
  ModuleSimpleHydro(const ModuleBuildInfo& cb);
  ~ModuleSimpleHydro(); //!< Destructeur

 public:
  
  static void staticInitialize(ISubDomain* sd);

 public:
	
  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,1); }

 public:


  void hydroBuild();
  void hydroStartInit();
  void hydroInit();
  void hydroContinueInit(){}
  void hydroExit();

  void computeForces();
  void computePressureForce(){}
  void computePseudoViscosity(){}
  void computeVelocity();
  void computeViscosityWork();
  void applyBoundaryCondition();
  void moveNodes();
  void computeGeometricValues();
  void updateDensity();
  void applyEquationOfState();
  void computeDeltaT();
  void computeSecondaryVariables();
  void doOneIteration(){ ARCANE_THROW(NotImplementedException,""); }
  void onMeshChanged();

 private:
  
  void computeGeometricValues2();

  void cellScalarPseudoViscosity();
  inline void computeCQs(Real3 node_coord[8],Real3 face_coord[6],const Cell& cell);

 private:
  VariableScalarReal m_density_ratio_maximum; //!< Accroissement maximum de la densité sur un pas de temps
  VariableScalarReal m_delta_t_n; //!< Delta t n entre t^{n-1/2} et t^{n+1/2}
  VariableScalarReal m_delta_t_f; //!< Delta t n+\demi  entre t^{n} et t^{n+1}
  VariableScalarReal m_old_dt_f; //!< Delta t n-\demi  entre t^{n-1} et t^{n}

  SecondaryVariables* m_secondary_variables;
  bool m_is_backward_done;
  ITraceMessageListener* m_listing_listener;
 private:

  static void _createTimeLoop(ISubDomain* sd,Integer number);
  void _computePressureAndCellPseudoViscosityForces();
  void _specialInit();
  void _initEquationOfState();
 public:
  void _computeViscosityWork(CellVectorView cells);
  void _computeGeometricValues(CellVectorView view);
  void _applyEquationOfState(CellVectorView cells);

  void _checkGoBackward();
  IPrimaryMesh* _createMesh2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleSimpleHydro::
ModuleSimpleHydro(const ModuleBuildInfo& mb)
: ArcaneSimpleHydroObject(mb)
, m_density_ratio_maximum(VariableBuildInfo(this,"DensityRatioMaximum"))
, m_delta_t_n(VariableBuildInfo(this,"CenteredDeltaT"))
, m_delta_t_f(VariableBuildInfo(this,"SplitDeltaT"))
, m_old_dt_f(VariableBuildInfo(this,"OldDTf"))
, m_secondary_variables(0)
, m_is_backward_done(false)
, m_listing_listener(0)
{
  addEntryPoint(this,"SH_ComputeSecondaryVariables",
	  &ModuleSimpleHydro::computeSecondaryVariables);

  if (TaskFactory::isActive()){
    info() << "USE CONCURRENCY!!!!!";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
staticInitialize(ISubDomain*)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleSimpleHydro::
~ModuleSimpleHydro()
{
  delete m_listing_listener;
  delete m_secondary_variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* ModuleSimpleHydro::
_createMesh2()
{
  info() << "CREATE MESH2";
  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();
  IPrimaryMesh* new_mesh = sd->application()->mainFactory()->createMesh(sd,pm->sequentialParallelMng(),"Mesh2");
  new_mesh->setDimension(2);
  new_mesh->allocateCells(0,Int64ConstArrayView(),false);
  new_mesh->endAllocate();
  return new_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
hydroBuild()
{
  IPrimaryMesh* mesh2 = _createMesh2();
  subDomain()->meshMng()->destroyMesh(mesh2->handle());
  _createMesh2();

  info() << "Hydro build entry point";
  if (options()->specificTraceListener()){
    info() << "Utilise ITraceMessageListener";
    ITraceMessageListener* tp = new ArcaneTest::TestTraceMessageListener();
    m_listing_listener = tp;
    traceMng()->addListener(tp);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
hydroExit()
{
  info() << "Hydro exit entry point";
  if (m_listing_listener)
    traceMng()->removeListener(m_listing_listener);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialisation du module hydro lors du démarrage du cas.
 */
void ModuleSimpleHydro::
hydroStartInit()
{
  ProfilingRegistry::setProfilingLevel(2);

#if 0
  {
    Integer index = 0;
    ENUMERATE_NODE(inode,allNodes()){
      if (index>10)
        break;
      info() << " NODE MASS=" << m_node_mass[inode];
      ++index;
    }
  }
#endif

  mesh_utils::shrinkMeshGroups(mesh());
  allCells().internal()->checkLocalIdsAreContigous();
  allNodes().internal()->checkLocalIdsAreContigous();

  info() << "ALL_CELLS contigous?=" << allCells().internal()->isContigousLocalIds();
  info() << "ALL_NODES contigous?=" << allNodes().internal()->isContigousLocalIds();
  {
    Int32 rank = parallelMng()->commRank();
    ENUMERATE_(Cell,icell,allCells()){
      m_sub_domain_id[icell] = rank;
      m_rank_as_int16[icell] = (Int16)rank;
      m_cell_unique_id[icell] = icell->uniqueId();
    }
  }

  // A Activer pour tester le retout-arriere
  //subDomain()->timeLoopMng()->setBackwardSavePeriod(5);

  //_specialInit();

  // Dimensionne les variables tableaux
  m_cell_cqs.resize(8);

  // Vérifie que les valeurs initiales sont correctes
  {
    Integer nb_error = 0;
    ENUMERATE_(Cell,icell,allCells()){
      Real pressure = m_pressure[icell];
      Real adiabatic_cst = m_adiabatic_cst[icell];
      Real density = m_density[icell];
      if (math::isZero(pressure) || math::isZero(density) || math::isZero(adiabatic_cst)){
        info() << "Null valeur for cell=" << ItemPrinter(*icell)
               << " density=" << density
               << " pressure=" << pressure
               << " adiabatic_cst=" << adiabatic_cst;
        ++nb_error;
      }
    }
    if (nb_error!=0)
      fatal() << "Some (" << nb_error << ") cells are not initialised";
  }
  
  // Initialise le delta-t
  Real deltat_init = options()->deltatInit.value();
  m_delta_t_n = deltat_init;
  m_delta_t_f = deltat_init;

  // Initialise les données géométriques: volume, cqs, longueurs caractéristiques
  computeGeometricValues();

  m_node_mass.fill(ARCANE_REAL(0.0));
  m_velocity.fill(Real3::zero());

  // Initialisation de la masses des mailles et des masses nodale
  ENUMERATE_(Cell,icell,allCells()){
    Cell cell = *icell;
    m_cell_mass[icell] = m_density[icell] * m_volume[icell];

    Real contrib_node_mass = ARCANE_REAL(0.125) * m_cell_mass[cell];
    for( NodeLocalId i_node : cell.nodeIds() )
      m_node_mass[i_node] += contrib_node_mass;
  }

  m_node_mass.synchronize();

  _initEquationOfState();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
_initEquationOfState()
{
  auto in_pressure = viewIn(m_pressure);
  auto in_adiabatic_cst = viewIn(m_adiabatic_cst);
  auto in_density = viewIn(m_density);
  auto out_internal_energy = viewOut(m_internal_energy);
  auto out_sound_speed = viewOut(m_sound_speed);
  // Initialise l'énergie et la vitesse du son
  ENUMERATE_ITEM_LAMBDA(Cell,icell,allCells()){
    Real pressure = in_pressure[icell];
    Real adiabatic_cst = in_adiabatic_cst[icell];
    Real density = in_density[icell];
    out_internal_energy[icell] = pressure / ((adiabatic_cst-ARCANE_REAL(1.0)) * density);
    out_sound_speed[icell] = math::sqrt(adiabatic_cst*pressure/density);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
_checkGoBackward()
{
  if (m_is_backward_done)
    return;
  Int32 backward_iter = options()->backwardIteration();
  if (backward_iter==0){
    m_is_backward_done = true;
    return;
  }
  m_secondary_variables->m_unused_variable.setUsed(false);
  Int32 global_iter = m_global_iteration();
  if (global_iter>=backward_iter){
    m_is_backward_done = true;
    //subDomain()->timeLoopMng()->goBackward();
    throw GoBackwardException(A_FUNCINFO, "Test go backward"); 
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des forces au temps courant \f$t^{n}\f$
 */
void ModuleSimpleHydro::
computeForces()
{
  _checkGoBackward();

  // Remise à zéro du vecteur des forces.
  m_force.fill(Real3::null());

  // Calcul pour chaque noeud de chaque maille la contribution
  // des forces de pression et de la pseudo-viscosite si necessaire
  if (options()->viscosity()==TypesSimpleHydro::ViscosityCellScalar){
    _computePressureAndCellPseudoViscosityForces();
  }
  else{
    ENUMERATE_(Cell,icell,allCells()){
      Cell cell = *icell;
      Real pressure = m_pressure[cell];
      ENUMERATE_CONNECTED_(Node,i_node,cell,nodes())
        m_force[i_node] += pressure * m_cell_cqs[icell][i_node.index()];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Pseudo viscosité scalaire aux mailles
 */
void ModuleSimpleHydro::
_computePressureAndCellPseudoViscosityForces()
{
  Real linear_coef = options()->viscosityLinearCoef.value();
  Real quadratic_coef = options()->viscosityQuadraticCoef.value();
  // Boucle sur les mailles du maillage
  ENUMERATE_(Cell,icell,allCells().view()){
    Cell cell = *icell;
    //const Integer cell_nb_node = cell.nbNode();
    const Real rho = m_density[icell];
    const Real pressure = m_pressure[icell];

    // Calcul de la divergence de la vitesse
    Real delta_speed = 0.;
    for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node )
      delta_speed += math::scaMul(m_velocity[i_node],m_cell_cqs[icell][i_node.index()]);
    delta_speed /= m_volume[icell];

    // Capture uniquement les chocs
    bool shock = (math::min(ARCANE_REAL(0.0),delta_speed)<ARCANE_REAL(0.0));
    if (shock){
      Real sound_speed = m_sound_speed[icell];
      Real dx = m_caracteristic_length[icell];
      Real quadratic_viscosity = rho * dx * dx * delta_speed * delta_speed;
      Real linear_viscosity = -rho*sound_speed* dx * delta_speed;
      Real scalar_viscosity = linear_coef * linear_viscosity + quadratic_coef * quadratic_viscosity;
      m_cell_viscosity_force[icell] = scalar_viscosity;
      for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node )
        m_force[i_node] += (pressure + scalar_viscosity)*m_cell_cqs[icell][i_node.index()];
    }
    else{
      m_cell_viscosity_force[icell] = ARCANE_REAL(0.0);
      for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node )
        m_force[i_node] += pressure * m_cell_cqs[icell][i_node.index()];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase2).
 */
void ModuleSimpleHydro::
computeVelocity()
{
  m_force.synchronize();

  Real delta_t_n = m_delta_t_n();

  auto in_node_mass = viewIn(m_node_mass);
  auto in_force = viewIn(m_force);
  auto inout_velocity = viewInOut(m_velocity);

  // Calcule l'impulsion aux noeuds
  ENUMERATE_ITEM_LAMBDA(Node,inode,allNodes()){
     Real node_mass  = in_node_mass[inode];

    Real3 old_velocity = inout_velocity[inode];
    Real3 new_velocity = old_velocity + (delta_t_n / node_mass) * in_force[inode];

    inout_velocity[inode] = new_velocity;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase3).
 */
void ModuleSimpleHydro::
computeViscosityWork()
{
  ForLoopRunInfo fri(ForLoopTraceInfo(A_FUNCINFO,"computeViscosityWork"));
  arcaneParallelForeach(allCells(),fri,this,&ModuleSimpleHydro::_computeViscosityWork);
}

void ModuleSimpleHydro::
_computeViscosityWork(CellVectorView cells)
{
  // Calcul du travail des forces de viscosité dans une maille
  ENUMERATE_(Cell,icell,cells){
    Cell cell = *icell;
    Real work = 0.;
    Real scalar_viscosity = m_cell_viscosity_force[icell];
    if (!math::isZero(scalar_viscosity))
      for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node )
        work += math::scaMul(scalar_viscosity*m_cell_cqs[icell][i_node.index()],m_velocity[i_node]);
    m_cell_viscosity_work[icell] = work;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prise en compte des conditions aux limites.
 */
void ModuleSimpleHydro::
applyBoundaryCondition()
{
  for( Integer i=0, nb=options()->boundaryCondition.count(); i<nb; ++i){
    FaceGroup face_group = options()->boundaryCondition[i].surface.value();
    NodeGroup node_group = face_group.nodeGroup();
    Real value = options()->boundaryCondition[i].value.value();
    TypesSimpleHydro::eBoundaryCondition type = options()->boundaryCondition[i].type.value();

    // boucle sur les faces de la surface
    ENUMERATE_(Face,iface,face_group){
      Face face = *iface;
      // boucle sur les noeuds de la face
      for( NodeLocalId node : face.nodeIds() ){
        switch(type) {
        case TypesSimpleHydro::VelocityX: m_velocity[node].x = value; break;
        case TypesSimpleHydro::VelocityY: m_velocity[node].y = value; break;
        case TypesSimpleHydro::VelocityZ: m_velocity[node].z = value; break;
        case TypesSimpleHydro::Unknown: break;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Déplace les noeuds.
 */
void ModuleSimpleHydro::
moveNodes()
{
  Real deltat_f = m_delta_t_f();
  
  ARCANE_HYODA_SOFTBREAK(subDomain());

  auto node_coord = viewInOut(m_node_coord);
  auto velocity = viewIn(m_velocity);
  ENUMERATE_ITEM_LAMBDA(Node,inode,allNodes()){
    node_coord[inode] += deltat_f * velocity[inode];
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mise à jour des densités et calcul de l'accroissements max
 *	  de la densité sur l'ensemble du maillage.
 */
void ModuleSimpleHydro::
updateDensity()
{
  Real density_ratio_maximum = ARCANE_REAL(0.0);
  
  ARCANE_HYODA_SOFTBREAK(subDomain());
    
  ENUMERATE_(Cell,icell,allCells()){
    Real old_density = m_density[icell];
    Real new_density = m_cell_mass[icell] / m_volume[icell];

    m_density[icell] = new_density;

    Real density_ratio = (new_density - old_density) / new_density;

    if (density_ratio_maximum<density_ratio)
      density_ratio_maximum = density_ratio;
  }

  m_density_ratio_maximum = density_ratio_maximum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'équation d'état et calcul l'énergie interne et la
 * pression.
 */
void ModuleSimpleHydro::
applyEquationOfState()
{
  ForLoopRunInfo fri(ForLoopTraceInfo(A_FUNCINFO,"applyEquationOfState"));
  arcaneParallelForeach(allCells(),fri,this,&ModuleSimpleHydro::_applyEquationOfState);
}

void ModuleSimpleHydro::
_applyEquationOfState(CellVectorView cells)
{
  const Real deltatf = m_delta_t_f();
  
  const bool add_viscosity_force = (options()->viscosity()!=TypesSimpleHydro::ViscosityNo);

  auto in_adiabatic_cst = viewIn(m_adiabatic_cst);
  auto in_volume = viewIn(m_volume);
  auto in_old_volume = viewIn(m_old_volume);
  auto in_cell_viscosity_work = viewIn(m_cell_viscosity_work);
  auto in_cell_mass = viewIn(m_cell_mass);
  auto out_internal_energy = viewOut(m_internal_energy);

    // Calcul de l'énergie interne
  ENUMERATE_ITEM_LAMBDA(Cell,icell,cells){
    Real adiabatic_cst = in_adiabatic_cst[icell];
    Real volume_ratio = in_volume[icell] / in_old_volume[icell];
    Real x = ARCANE_REAL(0.5)*(adiabatic_cst-ARCANE_REAL(1.0));
    Real numer_accrois_nrj = ARCANE_REAL(1.0) + x*(ARCANE_REAL(1.0)-volume_ratio);
    Real denom_accrois_nrj = ARCANE_REAL(1.0) + x*(ARCANE_REAL(1.0)-(ARCANE_REAL(1.0)/volume_ratio));
    /*info() << "RATIO " << ItemPrinter(*icell) << " n=" << numer_accrois_nrj
           << " d=" << denom_accrois_nrj << " volume_ratio=" << volume_ratio
           << " inv=" << (1.0/volume_ratio) << " x=" << x
           << " denom2=" << denom2 << " denom3=" << denom3 << " denom4=" << denom4;*/
    out_internal_energy[icell] *= numer_accrois_nrj/denom_accrois_nrj;
  
    // Prise en compte du travail des forces de viscosité 
    if (add_viscosity_force)
      out_internal_energy[icell] -= deltatf*in_cell_viscosity_work[icell] /
      (in_cell_mass[icell]*denom_accrois_nrj);
  };

  {
    auto in_internal_energy = viewIn(m_internal_energy);
    auto in_density = viewIn(m_density);
    auto in_adiabatic_cst = viewIn(m_adiabatic_cst);
    auto out_pressure = viewOut(m_pressure);
    auto out_sound_speed = viewOut(m_sound_speed);

    // Calcul de la pression et de la vitesse du son
    ENUMERATE_ITEM_LAMBDA(Cell,icell,cells){
      Real internal_energy = in_internal_energy[icell];
      Real density = in_density[icell];
      Real adiabatic_cst = in_adiabatic_cst[icell];
      Real pressure = (adiabatic_cst-ARCANE_REAL(1.0)) * density * internal_energy; 
      out_pressure[icell] = pressure;
      Real sound_speed = math::sqrt(adiabatic_cst*pressure/density);
      out_sound_speed[icell] = sound_speed;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des nouveaux pas de temps.
 */
void ModuleSimpleHydro::
computeDeltaT()
{
  {
    ITimeHistoryMng* thm = subDomain()->timeHistoryMng();
    Real v = 0.0;
    ENUMERATE_(Cell,icell,allCells()){
      v += m_pressure[icell] * m_density[icell];
    }
    thm->addValue("MyCurve",v);    
  }

  const Real old_dt = m_global_deltat();

  // Calcul du pas de temps pour le respect du critère de CFL
  
  Real minimum_aux = FloatInfo<Real>::maxValue();
  Real new_dt = FloatInfo<Real>::maxValue();

  ARCANE_PRAGMA_IVDEP
  ENUMERATE_(Cell,icell,ownCells()){
    Real cell_dx = m_caracteristic_length[icell];
    Real sound_speed = m_sound_speed[icell];
    Real dx_sound = cell_dx / sound_speed;
    minimum_aux = math::min(minimum_aux,dx_sound);
  };

  new_dt = options()->cfl()*minimum_aux;

  //Real cfl_dt = new_dt;

  // Pas de variations trop brutales à la hausse comme à la baisse
  Real max_dt = (ARCANE_REAL(1.0)+options()->variationSup())*old_dt;
  Real min_dt = (ARCANE_REAL(1.0)-options()->variationInf())*old_dt;

  new_dt = math::min(new_dt,max_dt);
  new_dt = math::max(new_dt,min_dt);

  //Real variation_min_max_dt = new_dt;

  // control de l'accroissement relatif de la densité
  Real dgr = options()->densityGlobalRatio();
  if (m_density_ratio_maximum()>dgr)
    new_dt = math::min(old_dt*dgr/m_density_ratio_maximum(),new_dt);

  new_dt = parallelMng()->reduce(Parallel::ReduceMin,new_dt);

  //Real density_ratio_dt = new_dt;

  // respect des valeurs min et max imposées par le fichier de données .plt
  new_dt = math::min(new_dt,options()->deltatMax());
  new_dt = math::max(new_dt,options()->deltatMin());

  //Real data_min_max_dt = new_dt;

  // Le dernier calcul se fait exactement au temps stopTime()
  {
    Real stop_time  = options()->finalTime();
    bool not_yet_finish = ( m_global_time() < stop_time);
    bool too_much = ( (m_global_time()+new_dt) > stop_time);

    if ( not_yet_finish && too_much ){
      new_dt = stop_time - m_global_time();
      subDomain()->timeLoopMng()->stopComputeLoop(true);
    }
  }

  // Mise à jour des variables
  m_old_dt_f.assign(old_dt);
  m_delta_t_n.assign(ARCANE_REAL(0.5)*(old_dt+new_dt));
  m_delta_t_f.assign(new_dt);
  m_global_deltat.assign(new_dt);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des résultantes aux noeuds d'une maille hexaédrique.
 *
 * La méthode utilisée est celle du découpage en quatre triangles.
 */
inline void ModuleSimpleHydro::
computeCQs(Real3 node_coord[8],Real3 face_coord[6],const Cell& cell)
{
  const Real3 c0 = face_coord[0];
  const Real3 c1 = face_coord[1];
  const Real3 c2 = face_coord[2];
  const Real3 c3 = face_coord[3];
  const Real3 c4 = face_coord[4];
  const Real3 c5 = face_coord[5];

  const Real demi = ARCANE_REAL(0.5);
  const Real five = ARCANE_REAL(5.0);

  // Calcul des normales face 1 :
  const Real3 n1a04 = demi * math::cross(node_coord[0] - c0 , node_coord[3] - c0);
  const Real3 n1a03 = demi * math::cross(node_coord[3] - c0 , node_coord[2] - c0);
  const Real3 n1a02 = demi * math::cross(node_coord[2] - c0 , node_coord[1] - c0);
  const Real3 n1a01 = demi * math::cross(node_coord[1] - c0 , node_coord[0] - c0);

  // Calcul des normales face 2 :
  const Real3 n2a05 = demi * math::cross(node_coord[0] - c1 , node_coord[4] - c1);
  const Real3 n2a12 = demi * math::cross(node_coord[4] - c1 , node_coord[7] - c1);
  const Real3 n2a08 = demi * math::cross(node_coord[7] - c1 , node_coord[3] - c1);
  const Real3 n2a04 = demi * math::cross(node_coord[3] - c1 , node_coord[0] - c1);

  // Calcul des normales face 3 :
  const Real3 n3a01 = demi * math::cross(node_coord[0] - c2 , node_coord[1] - c2);
  const Real3 n3a06 = demi * math::cross(node_coord[1] - c2 , node_coord[5] - c2);
  const Real3 n3a09 = demi * math::cross(node_coord[5] - c2 , node_coord[4] - c2);
  const Real3 n3a05 = demi * math::cross(node_coord[4] - c2 , node_coord[0] - c2);

  // Calcul des normales face 4 :
  const Real3 n4a09 = demi * math::cross(node_coord[4] - c3 , node_coord[5] - c3);
  const Real3 n4a10 = demi * math::cross(node_coord[5] - c3 , node_coord[6] - c3);
  const Real3 n4a11 = demi * math::cross(node_coord[6] - c3 , node_coord[7] - c3);
  const Real3 n4a12 = demi * math::cross(node_coord[7] - c3 , node_coord[4] - c3);
	
  // Calcul des normales face 5 :
  const Real3 n5a02 = demi * math::cross(node_coord[1] - c4 , node_coord[2] - c4);
  const Real3 n5a07 = demi * math::cross(node_coord[2] - c4 , node_coord[6] - c4);
  const Real3 n5a10 = demi * math::cross(node_coord[6] - c4 , node_coord[5] - c4);
  const Real3 n5a06 = demi * math::cross(node_coord[5] - c4 , node_coord[1] - c4);
      
  // Calcul des normales face 6 :
  const Real3 n6a03 = demi * math::cross(node_coord[2] - c5 , node_coord[3] - c5);
  const Real3 n6a08 = demi * math::cross(node_coord[3] - c5 , node_coord[7] - c5);
  const Real3 n6a11 = demi * math::cross(node_coord[7] - c5 , node_coord[6] - c5);
  const Real3 n6a07 = demi * math::cross(node_coord[6] - c5 , node_coord[2] - c5);

  const Real real_1div12 = ARCANE_REAL(1.0) / ARCANE_REAL(12.0);

  // Calcul des résultantes aux sommets :
  m_cell_cqs(cell,0) = (five*(n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
                        (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09))*real_1div12;
  m_cell_cqs(cell,1) = (five*(n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
                        (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07))*real_1div12;
  m_cell_cqs(cell,2) = (five*(n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
                        (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08))*real_1div12;
  m_cell_cqs(cell,3) = (five*(n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
                        (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11))*real_1div12;
  m_cell_cqs(cell,4) = (five*(n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
                        (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11))*real_1div12;
  m_cell_cqs(cell,5) = (five*(n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
                        (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02))*real_1div12;
  m_cell_cqs(cell,6) = (five*(n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
                        (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08))*real_1div12;
  m_cell_cqs(cell,7) = (five*(n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
                        (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03))*real_1div12;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du volume des mailles, des longueurs caractéristiques
 * et des résultantes aux sommets.
 */
void ModuleSimpleHydro::
computeGeometricValues()
{
  ForLoopRunInfo fri(ForLoopTraceInfo(A_FUNCINFO,"computeGeometricValues"));
  arcaneParallelForeach(allCells(),fri,this,&ModuleSimpleHydro::_computeGeometricValues);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du volume des mailles, des longueurs caractéristiques
 * et des résultantes aux sommets.
 */
void ModuleSimpleHydro::
_computeGeometricValues(CellVectorView cells)
{
  // Copie locale des coordonnées des sommets d'une maille
  Real3 coord[8];
  // Coordonnées des centres des faces
  Real3 face_coord[6];

  ARCANE_PRAGMA_IVDEP
  ENUMERATE_(Cell,icell,cells){
    Cell cell = *icell;

    // Recopie les coordonnées locales (pour le cache)
    for( NodeEnumerator i_node(cell.nodes()); i_node.index()<8; ++i_node )
      coord[i_node.index()] = m_node_coord[i_node];

    // Calcul les coordonnées des centres des faces
    face_coord[0] = ARCANE_REAL(0.25) * ( coord[0] + coord[3] + coord[2] + coord[1] );
    face_coord[1] = ARCANE_REAL(0.25) * ( coord[0] + coord[4] + coord[7] + coord[3] );
    face_coord[2] = ARCANE_REAL(0.25) * ( coord[0] + coord[1] + coord[5] + coord[4] );
    face_coord[3] = ARCANE_REAL(0.25) * ( coord[4] + coord[5] + coord[6] + coord[7] );
    face_coord[4] = ARCANE_REAL(0.25) * ( coord[1] + coord[2] + coord[6] + coord[5] );
    face_coord[5] = ARCANE_REAL(0.25) * ( coord[2] + coord[3] + coord[7] + coord[6] );

    // Calcule la longueur caractéristique de la maille.
    {
      Real3 median1 = face_coord[0]-face_coord[3];
      Real3 median2 = face_coord[2]-face_coord[5];
      Real3 median3 = face_coord[1]-face_coord[4];
      Real d1 = median1.normL2();
      Real d2 = median2.normL2();
      Real d3 = median3.normL2();

      Real dx_numerator   = d1*d2*d3;
      Real dx_denominator = d1*d2 + d1*d3 + d2*d3;
      m_caracteristic_length[icell] = dx_numerator / dx_denominator;
    }

    // Calcule les résultantes aux sommets
    computeCQs(coord,face_coord,cell);

    // Calcule le volume de la maille
    {
      Real volume = 0.;
      for( Integer i_node=0; i_node<8; ++i_node )
        volume += math::dot(coord[i_node],m_cell_cqs(icell,i_node));
      volume /= ARCANE_REAL(3.0);

      m_old_volume(icell) = m_volume(icell);
      m_volume(icell) = volume;
    }

  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
hydroInit()
{
  if (options()->backwardIteration()!=0){
    subDomain()->timeLoopMng()->setBackwardSavePeriod(10);
    m_secondary_variables = new SecondaryVariables(this);
  }
  Real dt = m_global_deltat();
  Real dt_max = options()->deltatMax();
  info() << "INIT: DTmin=" << options()->deltatMin()
         << " DTmax="<< dt_max
         << " DT="<< dt;
  if (dt > dt_max)
    ARCANE_FATAL("DeltaT ({0}) > DTMax ({1})",dt,dt_max);

  if (subDomain()->isContinue()){
    if (platform::getEnvironmentVariable("ARCANE_CONTINUE_OLD_CHECKPOINT")!="1"){
      Int64 t = defaultMesh()->timestamp();
      info() << "HydroContinueInit: MeshTimeStamp=" << t;
      if (t==0)
        ARCANE_FATAL("Mesh timestamp should not be zero");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
computeSecondaryVariables()
{
  if (!m_secondary_variables)
    m_secondary_variables = new SecondaryVariables(this);
  
  ENUMERATE_(Face,i_face,allFaces()){
    Face face = *i_face;
    Real face_density = 0.;
    Integer nb_cell = face.nbCell();
    for( Integer i=0; i<nb_cell; ++i ){
      Cell cell = face.cell(i);
      face_density += m_density[cell];
    }
    if (nb_cell!=0)
      face_density /= static_cast<double>(nb_cell);
    m_secondary_variables->m_faces_density[face] = face_density;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydro::
_specialInit()
{
  // Récupère les positions zmin et zmax
  //FaceGroup zmin = findGroup("ZMIN");
  //FaceGroup zmax = findGroup("ZMAX");

  IMesh* mesh = defaultMesh();
  VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
  Real3 min_pos;
  Real3 max_pos;
  ENUMERATE_(Node,inode,mesh->allNodes()){
    Real3 pos = nodes_coord_var[inode];
    if (inode.index()==0){
      min_pos = pos;
      max_pos = pos;
      continue;
    }
    if (min_pos.x>pos.x)
      min_pos.x = pos.x;
    if (min_pos.y>pos.y)
      min_pos.y = pos.y;
    if (min_pos.z>pos.z)
      min_pos.z = pos.z;

    if (max_pos.x<pos.x)
      max_pos.x = pos.x;
    if (max_pos.y<pos.y)
      max_pos.y = pos.y;
    if (max_pos.z<pos.z)
      max_pos.z = pos.z;
  }
  IParallelMng* pm = mesh->parallelMng();
  info() << "MIN_POS=" << min_pos << " MAX_POS=" << max_pos;
  pm->reduce(Parallel::ReduceMin,RealArrayView(3,(Real*)&min_pos));
  pm->reduce(Parallel::ReduceMax,RealArrayView(3,(Real*)&max_pos));
  Real3 middle_pos = (min_pos + max_pos) * 0.5;
  info() << "GLOBAL_MIN_POS=" << min_pos << " GLOBAL_MAX_POS=" << max_pos;
  ENUMERATE_(Cell,icell,mesh->allCells()){
    Cell cell = *icell;
    Real3 p;
    for( NodeLocalId node_id : cell.nodeIds() )
      p += nodes_coord_var[node_id];
    p /= cell.nbNode();
    Real density = 1.0;
    Real pressure = 1.0;
    Real adiabatic_cst = 1.4;
    if (p.z < middle_pos.z){
      density = 0.125;
      pressure = 0.1;
      adiabatic_cst = 1.2;
    }
    if (p.x < middle_pos.x ){
      density = 10.0;
      pressure = 10.0;
      adiabatic_cst = 1.6;
    }
      
    m_density[icell] = density;
    m_pressure[icell] = pressure;
    m_adiabatic_cst[icell] = adiabatic_cst;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Point d'entrée appelé après un changement de maillage (par exemple
 * suite à un équilibrage de charge).
 *
 * Dans ce cas on remet à jour les informations sur le rang propriétaire des mailles.
 */
void ModuleSimpleHydro::
onMeshChanged()
{
  Int32 rank = parallelMng()->commRank();
  ENUMERATE_(Cell,icell,ownCells()){
    m_sub_domain_id[icell] = rank;
  }
  m_sub_domain_id.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(ModuleSimpleHydro,SimpleHydro);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace SimpleHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
