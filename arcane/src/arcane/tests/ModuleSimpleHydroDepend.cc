// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleSimpleHydroDepend.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Module Hydrodynamique simple utilisant les dépendances.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/BasicModule.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/MathUtils.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/VariableComputeFunction.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableDependInfo.h"
#include "arcane/core/IVariableUtilities.h"

#include "arcane/tests/TypesSimpleHydro.h"
#include "arcane/tests/SimpleHydro_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SimpleHydro
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module hydrodynamique simplifié avec dépendances.
 *
 * Ce module implémente une hydrodynamique simple tri-dimensionnel,
 * parallèle, avec une pseudo-viscosité aux mailles.
 *
 * Ce module utilise les mécanismes de dépendances des variables
 * \ref arcanedoc_core_types_axl_variable_depends.
 */
class ModuleSimpleHydroDepend
: public ArcaneSimpleHydroObject
{
 public:

  class SecondaryVariables
  {
   public:
    explicit SecondaryVariables(IModule* module)
    : m_faces_density(VariableBuilder(module,"FacesDensity"))
      {
        //m_faces_density.doInit(false);
      }
   public:
    VariableFaceReal m_faces_density;
  };

 public:

  //! Constructeur
  explicit ModuleSimpleHydroDepend(const ModuleBuilder& cb);
  ~ModuleSimpleHydroDepend(); //!< Destructeur

 public:
  
  static void staticInitialize(ISubDomain* sd);

 public:
	
  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,1); }

 public:


  void hydroStartInit();
  void hydroInit1();

  void applyEquationOfState();
  void computeDeltaT();
  void computeSecondaryVariables();

 public:
  
  void setForceDepend();
  void setViscosityForceDepend();
  void setLagrangianVelocityDepend();
  void setVelocityDepend();
  void setViscosityWorkDepend();
  void setNodeCoordDepend();
  void setDensityDepend();
  void setGeometricValueDepend();

  void computeForces();
  void computePseudoViscosity();
  void computeLagrangianVelocity();
  void computeVelocity();
  void computeViscosityWork();
  void moveNodes();
  void updateDensity();
  void computeGeometricValues();

  void hydroBuild(){}
  void hydroExit(){}
  void hydroInit(){}
  void hydroContinueInit(){}
  void applyBoundaryCondition(){}
  void doOneIteration(){ ARCANE_THROW(NotImplementedException,""); }
  void onMeshChanged(){}

 private:
  
  void computeGeometricValues2();

  void cellScalarPseudoViscosity();
  inline void computeCQs(Real3 node_coord[8],Real3 face_coord[6],Cell cell);

 private:

  VariableCellReal m_density; //!< Densite par maille
  VariableCellReal m_pressure; //!< Pression par maille
  VariableCellReal m_cell_mass; //!< Masse par maille
  VariableCellReal m_internal_energy;  //!< Energie interne des mailles
  VariableCellReal m_volume; //!< Volume des mailles
  VariableCellReal m_old_volume; //!< Volume d'une maille à l'itération précédente
  VariableNodeReal3 m_force;  //!< Force aux noeuds
  VariableNodeReal3 m_velocity; //!< Vitesse aux noeuds
  VariableNodeReal3 m_lagrangian_velocity; //!< Vitesse aux noeuds
  VariableNodeReal m_node_mass; //! Masse nodale
  VariableCellReal m_cell_viscosity_force;  //!< Contribution locale des forces de viscosité
  VariableCellReal m_viscosity_work;  //!< Travail des forces de viscosité par maille
  VariableCellReal m_adiabatic_cst; //!< Constante adiabatique par maille
  VariableCellReal m_caracteristic_length; //!< Longueur caractéristique par maille
  VariableCellReal m_sound_speed; //!< Vitesse du son dans la maille
  VariableNodeReal3 m_node_coord; //!< Coordonnées des noeuds
  VariableCellArrayReal3 m_cell_cqs; //!< Résultantes aux sommets pour chaque maille

  VariableScalarReal m_density_ratio_maximum; //!< Accroissement maximum de la densité sur un pas de temps
  VariableScalarReal m_delta_t_n; //!< Delta t n entre t^{n-1/2} et t^{n+1/2}
  VariableScalarReal m_delta_t_f; //!< Delta t n+\demi  entre t^{n} et t^{n+1}
  VariableScalarReal m_old_dt_f; //!< Delta t n-\demi  entre t^{n-1} et t^{n}

  SecondaryVariables* m_secondary_variables;

 private:

  static void _createTimeLoop(ISubDomain* sd,Integer number);
  IOnlineDebuggerService *hyoda;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(ModuleSimpleHydroDepend,SimpleHydroDepend);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleSimpleHydroDepend::
ModuleSimpleHydroDepend(const ModuleBuildInfo& mb)
: ArcaneSimpleHydroObject(mb)
, m_density (VariableBuilder(this,"Density"))
, m_pressure(VariableBuilder(this,"Pressure"))
, m_cell_mass(VariableBuilder(this,"CellMass"))  
, m_internal_energy(VariableBuilder(this,"InternalEnergy"))  
, m_volume(VariableBuilder(this,"CellVolume"))
, m_old_volume(VariableBuilder(this,"OldCellVolume"))
, m_force(VariableBuilder(this,"Force",IVariable::PNoDump|IVariable::PNoNeedSync))
, m_velocity(VariableBuilder(this,"Velocity"))
, m_lagrangian_velocity(VariableBuilder(this,"LagrangianVelocity"))
, m_node_mass(VariableBuilder(this,"NodeMass"))
, m_cell_viscosity_force(VariableBuilder(this,"CellViscosityForce",IVariable::PNoDump))
, m_viscosity_work(VariableBuilder(this,"ViscosityWork"))
, m_adiabatic_cst(VariableBuilder(this,"AdiabaticCst"))
, m_caracteristic_length(VariableBuilder(this,"CaracteristicLength",IVariable::PNoDump))
, m_sound_speed(VariableBuilder(this,"SoundSpeed"))
, m_node_coord(VariableBuilder(this,"NodeCoord"))
, m_cell_cqs(VariableBuilder(this,"CellCQS"))
, m_density_ratio_maximum(VariableBuilder(this,"DensityRatioMaximum"))
, m_delta_t_n(VariableBuilder(this,"CenteredDeltaT"))
, m_delta_t_f(VariableBuilder(this,"SplitDeltaT"))
, m_old_dt_f(VariableBuilder(this,"OldDTf"))
, m_secondary_variables(0)
, hyoda(platform::getOnlineDebuggerService())
{
  addEntryPoint(this,"SHD_HydroStartInit",
	  &ModuleSimpleHydroDepend::hydroStartInit,
	  IEntryPoint::WStartInit);
  addEntryPoint(this,"SHD_HydroInit1",
	  &ModuleSimpleHydroDepend::hydroInit1,
	  IEntryPoint::WInit);
  addEntryPoint(this,"SHD_ComputeGeometricValues",
                &ModuleSimpleHydroDepend::computeGeometricValues);
  addEntryPoint(this,"SHD_ApplyEquationOfState",
	  &ModuleSimpleHydroDepend::applyEquationOfState);
  addEntryPoint(this,"SHD_ComputeDeltaT",
	  &ModuleSimpleHydroDepend::computeDeltaT);

  addEntryPoint(this,"SHD_ComputeSecondaryVariables",
	  &ModuleSimpleHydroDepend::computeSecondaryVariables);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
_createTimeLoop(ISubDomain* sd,Integer number)
{
  String time_loop_name("HydroSimpleDepend");
  if (number>0)
    time_loop_name = time_loop_name + number;

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("SimpleHydroDepend.SHD_HydroStartInit"));
    clist.add(TimeLoopEntryPointInfo("SimpleHydroDepend.SHD_HydroInit1"));
    time_loop->setEntryPoints(ITimeLoop::WInit,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("SimpleHydroDepend.SHD_ComputeGeometricValues"));
    clist.add(TimeLoopEntryPointInfo("SimpleHydroDepend.SHD_ApplyEquationOfState"));
    if (number==1){
      clist.add(TimeLoopEntryPointInfo("SimpleHydroDepend.SHD_ComputeSecondaryVariables"));
    }
    clist.add(TimeLoopEntryPointInfo("SimpleHydroDepend.SHD_ComputeDeltaT"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop,clist);
  }

  {
    StringList clist;
    clist.add("SimpleHydroDepend");
    time_loop->setRequiredModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
staticInitialize(ISubDomain* sd)
{
  _createTimeLoop(sd,0);
  _createTimeLoop(sd,1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleSimpleHydroDepend::
~ModuleSimpleHydroDepend()
{
  delete m_secondary_variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialisation du module hydro lors du démarrage du cas.
 */
void ModuleSimpleHydroDepend::
hydroStartInit()
{
  // Dimensionne les variables tableaux
  m_cell_cqs.resize(8);

  // Initialise le delta-t
  Real deltat_init = options()->deltatInit();
  m_delta_t_n = deltat_init;
  m_delta_t_f = deltat_init;

  m_density.setUpToDate();
  m_pressure.setUpToDate();
  m_adiabatic_cst.setUpToDate();

  m_node_coord.setUpToDate();

  // Initialise les données géométriques: volume, cqs, longueurs caractéristiques
  computeGeometricValues();

  m_velocity.fill(Real3::zero());
  m_velocity.setUpToDate();

  // Initialisation de la masses des mailles et des masses nodale
  m_node_mass.fill(0.0);
  ENUMERATE_CELL(icell,allCells()){
    const Cell& cell = *icell;

    m_cell_mass[icell] = m_density[icell] * m_volume[icell];

    Real contrib_node_mass = 0.125 * m_cell_mass[cell];
    for( Integer i_node=0, nb_node=cell.nbNode(); i_node<nb_node; ++i_node){
      m_node_mass[cell.node(i_node)] += contrib_node_mass;
    }
  }
  
  m_node_mass.synchronize();
  m_node_mass.setUpToDate();

  // Initialise l'énergie et la vitesse du son
  ENUMERATE_CELL(icell,allCells()){
    Real pressure = m_pressure[icell];
    Real adiabatic_cst = m_adiabatic_cst[icell];
    Real density = m_density[icell];
    m_internal_energy[icell] = pressure / ((adiabatic_cst-1.) * density);
    m_sound_speed[icell] = math::sqrt(adiabatic_cst*pressure/density);
  }
  m_internal_energy.setUpToDate();
  m_sound_speed.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
hydroInit1()
{
  setForceDepend();
  setViscosityForceDepend();
  setLagrangianVelocityDepend();
  setVelocityDepend();
  setViscosityWorkDepend();
  setNodeCoordDepend();
  setDensityDepend();
  setGeometricValueDepend();

  // Affiche les infos sur les variables
  IVariableMng* vm = subDomain()->variableMng();
  {
    std::ostringstream ostr;
    vm->utilities()->dumpAllVariableDependencies(ostr,true);
    info() << "Recursive dependencies\n" << ostr.str();
  }
  {
    std::ostringstream ostr;
    vm->utilities()->dumpAllVariableDependencies(ostr,false);
    info() << "Simple dependencies\n" << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setForceDepend()
{
  m_force.addDependPreviousTime(m_cell_cqs,A_FUNCINFO);
  m_force.addDependPreviousTime(m_pressure,A_FUNCINFO);

  m_force.addDependCurrentTime(m_cell_viscosity_force,A_FUNCINFO);

  m_force.setComputeFunction(this,&ModuleSimpleHydroDepend::computeForces);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des forces de pression au temps courant \f$t^{n}\f$
 */
void ModuleSimpleHydroDepend::
computeForces()
{
  // Remise à zéro du vecteur des forces.
  m_force.fill(Real3::null());

  // Calcul pour chaque noeud de chaque maille la contribution
  // des forces de pression
  ENUMERATE_CELL(icell,allCells()){
    Cell cell = *icell;
    Real pressure = m_pressure[cell];
    for( Integer i_node=0, nb_node=cell.nbNode(); i_node<nb_node; ++i_node )
      m_force[cell.node(i_node)] += pressure * m_cell_cqs[icell][i_node];
  }

  // Prise en compte des forces de viscosité si demandé
  bool add_viscosity_force = (options()->viscosity()!=TypesSimpleHydro::ViscosityNo);
  if (add_viscosity_force){
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Real scalar_viscosity = m_cell_viscosity_force[icell];
      for( Integer i_node=0, nb_node=cell.nbNode(); i_node<nb_node; ++i_node )
        m_force[cell.node(i_node)] += scalar_viscosity*m_cell_cqs[icell][i_node];
    }
  }
  m_force.setUpToDate();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setViscosityForceDepend()
{
  m_cell_viscosity_force.addDependPreviousTime(m_density,A_FUNCINFO);
  m_cell_viscosity_force.addDependPreviousTime(m_velocity,A_FUNCINFO);
  m_cell_viscosity_force.addDependPreviousTime(m_cell_cqs,A_FUNCINFO);
  m_cell_viscosity_force.addDependPreviousTime(m_volume,A_FUNCINFO);
  m_cell_viscosity_force.addDependPreviousTime(m_sound_speed,A_FUNCINFO);
  m_cell_viscosity_force.addDependPreviousTime(m_caracteristic_length,A_FUNCINFO);

  m_cell_viscosity_force.setComputeFunction(this,&ModuleSimpleHydroDepend::computePseudoViscosity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prise en compte de la pseudo viscosité en fonction des options choisies
 */
void ModuleSimpleHydroDepend::
computePseudoViscosity()
{
  if (options()->viscosity()!=TypesSimpleHydro::ViscosityCellScalar)
    return;

  Real linear_coef = options()->viscosityLinearCoef();
  Real quadratic_coef = options()->viscosityQuadraticCoef();

  // Boucle sur les mailles du maillage
  ENUMERATE_CELL(icell,allCells()){
    const Cell& cell = *icell;
    const Integer cell_nb_node = cell.nbNode();
    const Real rho = m_density[icell];

    // Calcul de la divergence de la vitesse
    Real delta_speed = 0.;
    for( Integer i_node=0; i_node<cell_nb_node; ++i_node )
      delta_speed += math::scaMul(m_velocity[cell.node(i_node)],m_cell_cqs[icell][i_node]);
    delta_speed /= m_volume[icell];

    // Capture uniquement les chocs
    bool shock = (math::min(Real(0.),delta_speed)<0.);
    if (shock){
      Real sound_speed = m_sound_speed[icell];
      Real dx = m_caracteristic_length[icell];
      Real quadratic_viscosity = rho * dx * dx * delta_speed * delta_speed;
      Real linear_viscosity = -rho*sound_speed* dx * delta_speed;
      Real scalar_viscosity = linear_coef * linear_viscosity + quadratic_coef * quadratic_viscosity;
      m_cell_viscosity_force[icell] = scalar_viscosity;
    }
    else
      m_cell_viscosity_force[icell] = 0.;
  }
  m_cell_viscosity_force.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setLagrangianVelocityDepend()
{
  m_lagrangian_velocity.addDependPreviousTime(m_node_mass);
  m_lagrangian_velocity.addDependCurrentTime(m_force);

  m_lagrangian_velocity.setComputeFunction(this,&ModuleSimpleHydroDepend::computeLagrangianVelocity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase2).
 */
void ModuleSimpleHydroDepend::
computeLagrangianVelocity()
{
  ENUMERATE_NODE(i_node,ownNodes()){
    Real node_mass  = m_node_mass[i_node];

    Real3 old_velocity = m_velocity[i_node];
    Real3 new_velocity = old_velocity + (m_delta_t_n() / node_mass) * m_force[i_node];

    m_lagrangian_velocity[i_node] = new_velocity;
  }

  m_lagrangian_velocity.setUpToDate();
  m_lagrangian_velocity.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setVelocityDepend()
{
  m_velocity.addDependCurrentTime(m_lagrangian_velocity);

  m_velocity.setComputeFunction(this,&ModuleSimpleHydroDepend::computeVelocity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
computeVelocity()
{
  m_velocity.copy(m_lagrangian_velocity);

  // Prise en compte des conditions aux limites.
  for( Integer i=0, nb=options()->boundaryCondition.size(); i<nb; ++i){
    FaceGroup face_group = options()->boundaryCondition[i]->surface();
    Real value = options()->boundaryCondition[i]->value();
    TypesSimpleHydro::eBoundaryCondition type = options()->boundaryCondition[i]->type();

    // boucle sur les faces de la surface
    ENUMERATE_FACE(j,face_group){
      const Face& face = *j;
      Integer nb_node = face.nbNode();				

      // boucle sur les noeuds de la face
      for( Integer k=0; k<nb_node; ++k ){
        const Node& node = face.node(k);
        switch(type) {
        case TypesSimpleHydro::VelocityX: m_velocity[node].x = value; break;
        case TypesSimpleHydro::VelocityY: m_velocity[node].y = value; break;
        case TypesSimpleHydro::VelocityZ: m_velocity[node].z = value; break;
        case TypesSimpleHydro::Unknown: break;
        }
      }
    }
  }

  m_velocity.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setViscosityWorkDepend()
{
  m_viscosity_work.addDependCurrentTime(m_lagrangian_velocity);
  m_viscosity_work.addDependCurrentTime(m_cell_viscosity_force);
  m_viscosity_work.addDependPreviousTime(m_cell_cqs);

  m_viscosity_work.setComputeFunction(this,&ModuleSimpleHydroDepend::computeViscosityWork);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase3).
 */
void ModuleSimpleHydroDepend::
computeViscosityWork()
{
  // Calcul du travail des forces de viscosité dans une maille
  ENUMERATE_CELL(icell,allCells()){
    Cell cell = *icell; 
    Real work = 0.;
    Real scalar_viscosity = m_cell_viscosity_force[icell];
    for( Integer i=0, n=cell.nbNode(); i<n; ++i )
      work += math::scaMul(scalar_viscosity*m_cell_cqs[icell][i],m_lagrangian_velocity[cell.node(i)]);
    m_viscosity_work[icell] = work;
  }
  m_viscosity_work.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setNodeCoordDepend()
{
  m_node_coord.addDependCurrentTime(m_velocity);

  m_node_coord.setComputeFunction(this,&ModuleSimpleHydroDepend::moveNodes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Déplace les noeuds.
 */
void ModuleSimpleHydroDepend::
moveNodes()
{
  Real deltat_f = m_delta_t_f();

  ENUMERATE_NODE(i_node,allNodes()){
    m_node_coord[i_node] += deltat_f * m_velocity[i_node];
  }

  m_node_coord.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setDensityDepend()
{
  m_density.addDependCurrentTime(m_volume);
  m_density.addDependCurrentTime(m_cell_mass);

  m_density.setComputeFunction(this,&ModuleSimpleHydroDepend::updateDensity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mise à jour des densités et calcul de l'accroissements max
 *	  de la densité sur l'ensemble du maillage.
 */
void ModuleSimpleHydroDepend::
updateDensity()
{
  Real density_ratio_maximum = 0.;

  ENUMERATE_CELL(icell,ownCells()){
    Real old_density = m_density[icell];
    Real new_density = m_cell_mass[icell] / m_volume[icell];

    m_density[icell] = new_density;

    Real density_ratio = (new_density - old_density) / new_density;

    if (density_ratio_maximum<density_ratio)
      density_ratio_maximum = density_ratio;
  }

  m_density_ratio_maximum = density_ratio_maximum;
  m_density_ratio_maximum.reduce(Parallel::ReduceMax);

  m_density_ratio_maximum.setUpToDate();

  m_density.synchronize();
  m_density.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'équation d'état et calcul l'énergie interne et la
 * pression.
 */
void ModuleSimpleHydroDepend::
applyEquationOfState()
{
  m_density.update();
  const Real deltatf = m_delta_t_f();
  
  const bool add_viscosity_force = (options()->viscosity()!=TypesSimpleHydro::ViscosityNo);
  if (add_viscosity_force){
    m_viscosity_work.update();
  }

  // Calcul de l'énergie interne
  ENUMERATE_CELL(icell,ownCells()){
    Real adiabatic_cst = m_adiabatic_cst[icell];
    Real volume_ratio = m_volume[icell] / m_old_volume[icell];
    Real x = 0.5*(adiabatic_cst-1.);
    Real numer_accrois_nrj = 1. + x*(1.-volume_ratio);
    Real denom_accrois_nrj = 1. + x*(1.-1./volume_ratio);

    m_internal_energy[icell] *= numer_accrois_nrj/denom_accrois_nrj;
  
    // Prise en compte du travail des forces de viscosité 
    if (add_viscosity_force)
      m_internal_energy[icell] -= deltatf*m_viscosity_work[icell] / 
	(m_cell_mass[icell]*denom_accrois_nrj);
  }

  // Synchronise l'énergie
  m_internal_energy.synchronize();

  // Calcul de la pression et de la vitesse du son
  ENUMERATE_CELL(icell,allCells()){
    Real internal_energy = m_internal_energy[icell];
    Real density = m_density[icell];
    Real adiabatic_cst = m_adiabatic_cst[icell];
    Real pressure = (adiabatic_cst-1.) * density * internal_energy; 
    m_pressure[icell] = pressure;
    m_sound_speed[icell] = math::sqrt(adiabatic_cst*pressure/density);
  }
  m_sound_speed.setUpToDate();
  m_internal_energy.setUpToDate();
  m_pressure.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des nouveaux pas de temps.
 */
void ModuleSimpleHydroDepend::
computeDeltaT()
{
  const Real old_dt = m_global_deltat();

  // Calcul du pas de temps pour le respect du critère de CFL
  
  Real minimum_aux = FloatInfo<Real>::maxValue();
  Real new_dt = FloatInfo<Real>::maxValue();

  ENUMERATE_CELL(icell,ownCells()){
    Real cell_dx = m_caracteristic_length[icell];
    Real sound_speed = m_sound_speed[icell];
    Real dx_sound = cell_dx / sound_speed;
    minimum_aux = math::min(minimum_aux,dx_sound);
  }

  new_dt = options()->cfl()*minimum_aux;

  // Pas de variations trop brutales à la hausse comme à la baisse
  Real max_dt = (1.+options()->variationSup())*old_dt;
  Real min_dt = (1.-options()->variationInf())*old_dt;

  new_dt = math::min(new_dt,max_dt);
  new_dt = math::max(new_dt,min_dt);

  // control de l'accroissement relatif de la densité
  Real dgr = options()->densityGlobalRatio();
  if (m_density_ratio_maximum()>dgr)
    new_dt = math::min(old_dt*dgr/m_density_ratio_maximum(),new_dt);

  new_dt = parallelMng()->reduce(Parallel::ReduceMin,new_dt);

  // respect des valeurs min et max imposées par le fichier de données .plt
  new_dt = math::min(new_dt,options()->deltatMax());
  new_dt = math::max(new_dt,options()->deltatMin());

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
  m_delta_t_n.assign(0.5*(old_dt+new_dt));
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
inline void ModuleSimpleHydroDepend::
computeCQs(Real3 node_coord[8],Real3 face_coord[6],Cell cell)
{
  const Real3 c0 = face_coord[0];
  const Real3 c1 = face_coord[1];
  const Real3 c2 = face_coord[2];
  const Real3 c3 = face_coord[3];
  const Real3 c4 = face_coord[4];
  const Real3 c5 = face_coord[5];

  // Calcul des normales face 1 :
  const Real3 n1a04 = 0.5 * math::vecMul(node_coord[0] - c0 , node_coord[3] - c0);
  const Real3 n1a03 = 0.5 * math::vecMul(node_coord[3] - c0 , node_coord[2] - c0);
  const Real3 n1a02 = 0.5 * math::vecMul(node_coord[2] - c0 , node_coord[1] - c0);
  const Real3 n1a01 = 0.5 * math::vecMul(node_coord[1] - c0 , node_coord[0] - c0);

  // Calcul des normales face 2 :
  const Real3 n2a05 = 0.5 * math::vecMul(node_coord[0] - c1 , node_coord[4] - c1);
  const Real3 n2a12 = 0.5 * math::vecMul(node_coord[4] - c1 , node_coord[7] - c1);
  const Real3 n2a08 = 0.5 * math::vecMul(node_coord[7] - c1 , node_coord[3] - c1);
  const Real3 n2a04 = 0.5 * math::vecMul(node_coord[3] - c1 , node_coord[0] - c1);

  // Calcul des normales face 3 :
  const Real3 n3a01 = 0.5 * math::vecMul(node_coord[0] - c2 , node_coord[1] - c2);
  const Real3 n3a06 = 0.5 * math::vecMul(node_coord[1] - c2 , node_coord[5] - c2);
  const Real3 n3a09 = 0.5 * math::vecMul(node_coord[5] - c2 , node_coord[4] - c2);
  const Real3 n3a05 = 0.5 * math::vecMul(node_coord[4] - c2 , node_coord[0] - c2);

  // Calcul des normales face 4 :
  const Real3 n4a09 = 0.5 * math::vecMul(node_coord[4] - c3 , node_coord[5] - c3);
  const Real3 n4a10 = 0.5 * math::vecMul(node_coord[5] - c3 , node_coord[6] - c3);
  const Real3 n4a11 = 0.5 * math::vecMul(node_coord[6] - c3 , node_coord[7] - c3);
  const Real3 n4a12 = 0.5 * math::vecMul(node_coord[7] - c3 , node_coord[4] - c3);
	
  // Calcul des normales face 5 :
  const Real3 n5a02 = 0.5 * math::vecMul(node_coord[1] - c4 , node_coord[2] - c4);
  const Real3 n5a07 = 0.5 * math::vecMul(node_coord[2] - c4 , node_coord[6] - c4);
  const Real3 n5a10 = 0.5 * math::vecMul(node_coord[6] - c4 , node_coord[5] - c4);
  const Real3 n5a06 = 0.5 * math::vecMul(node_coord[5] - c4 , node_coord[1] - c4);
      
  // Calcul des normales face 6 :
  const Real3 n6a03 = 0.5 * math::vecMul(node_coord[2] - c5 , node_coord[3] - c5);
  const Real3 n6a08 = 0.5 * math::vecMul(node_coord[3] - c5 , node_coord[7] - c5);
  const Real3 n6a11 = 0.5 * math::vecMul(node_coord[7] - c5 , node_coord[6] - c5);
  const Real3 n6a07 = 0.5 * math::vecMul(node_coord[6] - c5 , node_coord[2] - c5);

  // Calcul des résultantes aux sommets :
  m_cell_cqs[cell][0] = (5.*(n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
		 (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09))*(1./12.);
  m_cell_cqs[cell][1] = (5.*(n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
		 (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07))*(1./12.);
  m_cell_cqs[cell][2] = (5.*(n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
		 (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08))*(1./12.);
  m_cell_cqs[cell][3] = (5.*(n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
		 (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11))*(1./12.);
  m_cell_cqs[cell][4] = (5.*(n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
		 (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11))*(1./12.);
  m_cell_cqs[cell][5] = (5.*(n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
		 (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02))*(1./12.);
  m_cell_cqs[cell][6] = (5.*(n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
		 (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08))*(1./12.);
  m_cell_cqs[cell][7] = (5.*(n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
		 (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03))*(1./12.);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
setGeometricValueDepend()
{
  m_caracteristic_length.addDependCurrentTime(m_node_coord);
  m_caracteristic_length.setComputeFunction(this,&ModuleSimpleHydroDepend::computeGeometricValues);

  m_cell_cqs.addDependCurrentTime(m_node_coord);
  m_cell_cqs.setComputeFunction(this,&ModuleSimpleHydroDepend::computeGeometricValues);

  m_volume.addDependCurrentTime(m_node_coord);
  m_volume.setComputeFunction(this,&ModuleSimpleHydroDepend::computeGeometricValues);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du volume des mailles, des longueurs caractéristiques
 * et des résultantes aux sommets.
 */
void ModuleSimpleHydroDepend::
computeGeometricValues()
{
  m_old_volume.copy(m_volume);
  m_old_volume.setUpToDate();
  
  // Copie locale des coordonnées des sommets d'une maille
  Real3 coord[8];
  // Coordonnées des centres des faces
  Real3 face_coord[6];

  ENUMERATE_CELL(icell,allCells()){
    const Cell& cell = *icell;

    // Recopie les coordonnées locales (pour le cache)
    for( Integer i = 0; i<8; ++i )
      coord[i] = m_node_coord[cell.node(i)];

    // Calcul les coordonnées des centres des faces
    face_coord[0] = 0.25 * ( coord[0] + coord[3] + coord[2] + coord[1] );
    face_coord[1] = 0.25 * ( coord[0] + coord[4] + coord[7] + coord[3] );
    face_coord[2] = 0.25 * ( coord[0] + coord[1] + coord[5] + coord[4] );
    face_coord[3] = 0.25 * ( coord[4] + coord[5] + coord[6] + coord[7] );
    face_coord[4] = 0.25 * ( coord[1] + coord[2] + coord[6] + coord[5] );
    face_coord[5] = 0.25 * ( coord[2] + coord[3] + coord[7] + coord[6] );

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
      for( Integer i_node=0, nb_node=cell.nbNode(); i_node<nb_node; ++i_node )
        volume += math::scaMul(coord[i_node],m_cell_cqs[icell][i_node]);
      volume /= 3.0;

      m_volume[icell] = volume;
    }

  }
  m_cell_cqs.setUpToDate();
  m_volume.setUpToDate();
  m_caracteristic_length.setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleSimpleHydroDepend::
computeSecondaryVariables()
{
  if (!m_secondary_variables)
    m_secondary_variables = new SecondaryVariables(this);
  
  ENUMERATE_FACE(i_face,allFaces()){
    const Face& face = *i_face;
    Real face_density = 0.;
    Integer nb_cell = face.nbCell();
    for( Integer i=0; i<nb_cell; ++i ){
      const Cell& cell = face.cell(i);
      face_density += m_density[cell];
    }
    if (nb_cell!=0)
      face_density /= static_cast<double>(nb_cell);
    m_secondary_variables->m_faces_density[face] = face_density;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
