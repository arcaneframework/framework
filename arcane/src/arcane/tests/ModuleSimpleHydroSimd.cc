// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleSimpleHydroSimd.cc                                    (C) 2000-2020 */
/*                                                                           */
/* Module Hydrodynamique simple avec vectorisation.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#ifdef ARCANE_CHECK
#define ARCANE_TRACE_ENUMERATOR
#endif

#include "arcane/utils/List.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ITimeLoop.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/EntryPoint.h"
#include "arcane/MathUtils.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/VariableTypes.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IParallelMng.h"
#include "arcane/ModuleFactory.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/ItemPrinter.h"
#include "arcane/Concurrency.h"
#include "arcane/BasicService.h"
#include "arcane/ServiceBuildInfo.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/FactoryService.h"

#include "arcane/IMainFactory.h"

#include "arcane/tests/TypesSimpleHydro.h"

#include "arcane/SimdMathUtils.h"
#include "arcane/SimdItem.h"
#include "arcane/VariableView.h"

#ifdef __INTEL_COMPILER
#define HYDRO_PRAGMA_IVDEP _Pragma("ivdep")
#else
#define HYDRO_PRAGMA_IVDEP
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SimpleHydro
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module hydrodynamique simplifié avec vectorisation et parallélisation
 * par les threads. 
 *
 * Ce module implémente une hydrodynamique simple tri-dimensionnel,
 * parallèle, avec une pseudo-viscosité aux mailles en utilisant
 * les classes de vectorisation fournies par Arcane.
 */
class SimpleHydroSimdService
: public BasicService
, public ISimpleHydroService
{
 public:

  //! Constructeur
  explicit SimpleHydroSimdService(const ServiceBuildInfo& sbi);
  ~SimpleHydroSimdService(); //!< Destructeur

 public:
  
  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,1); }

 public:


  void hydroBuild() override;
  void hydroStartInit() override;
  void hydroInit() override;
  void hydroExit() override;

  void computeForces() override;
  void computeVelocity() override;
  void computeViscosityWork() override;
  void applyBoundaryCondition() override;
  void moveNodes() override;
  void computeGeometricValues() override;
  void updateDensity() override;
  void applyEquationOfState() override;
  void computeDeltaT() override;

  void setModule(SimpleHydro::SimpleHydroModuleBase* module) override
  {
    m_module = module;
  }

 private:
  
  void computeGeometricValues2();

  void cellScalarPseudoViscosity();
  inline void computeCQs(Real3 node_coord[8],Real3 face_coord[6],Cell cell);

 private:
  VariableCellInt64 m_cell_unique_id; //!< Unique ID associé à la maille
  VariableCellInt32 m_sub_domain_id; //!< Numéro du sous-domaine associé à la maille
  VariableCellReal m_density; //!< Densite par maille
  VariableCellReal m_pressure; //!< Pression par maille
  VariableCellReal m_cell_mass; //!< Masse par maille
  VariableCellReal m_internal_energy;  //!< Energie interne des mailles
  VariableCellReal m_volume; //!< Volume des mailles
  VariableCellReal m_old_volume; //!< Volume d'une maille à l'itération précédente
  VariableNodeReal3 m_force;  //!< Force aux noeuds
  VariableNodeReal3 m_velocity; //!< Vitesse aux noeuds
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

  SimpleHydro::SimpleHydroModuleBase* m_module;

 private:

  void _computePressureAndCellPseudoViscosityForces();
  void _specialInit();

 public:

  void computeCQsSimd(SimdReal3 node_coord[8],SimdReal3 face_coord[6],SimdReal3 cqs[8]);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHydroSimdService::
SimpleHydroSimdService(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_cell_unique_id (VariableBuildInfo(sbi.mesh(),"UniqueId"))
, m_sub_domain_id (VariableBuildInfo(sbi.mesh(),"SubDomainId"))
, m_density (VariableBuildInfo(sbi.mesh(),"Density"))
, m_pressure(VariableBuildInfo(sbi.mesh(),"Pressure"))
, m_cell_mass(VariableBuildInfo(sbi.mesh(),"CellMass"))  
, m_internal_energy(VariableBuildInfo(sbi.mesh(),"InternalEnergy"))  
, m_volume(VariableBuildInfo(sbi.mesh(),"CellVolume"))
, m_old_volume(VariableBuildInfo(sbi.mesh(),"OldCellVolume"))
, m_force(VariableBuildInfo(sbi.mesh(),"Force",IVariable::PNoNeedSync))
, m_velocity(VariableBuildInfo(sbi.mesh(),"Velocity"))
, m_node_mass(VariableBuildInfo(sbi.mesh(),"NodeMass"))
, m_cell_viscosity_force(VariableBuildInfo(sbi.mesh(),"CellViscosityForce"))
, m_viscosity_work(VariableBuildInfo(sbi.mesh(),"ViscosityWork"))
, m_adiabatic_cst(VariableBuildInfo(sbi.mesh(),"AdiabaticCst"))
, m_caracteristic_length(VariableBuildInfo(sbi.mesh(),"CaracteristicLength"))
, m_sound_speed(VariableBuildInfo(sbi.mesh(),"SoundSpeed"))
, m_node_coord(VariableBuildInfo(sbi.mesh(),"NodeCoord"))
, m_cell_cqs(VariableBuildInfo(sbi.mesh(),"CellCQS"))
, m_density_ratio_maximum(VariableBuildInfo(sbi.mesh(),"DensityRatioMaximum"))
, m_delta_t_n(VariableBuildInfo(sbi.mesh(),"CenteredDeltaT"))
, m_delta_t_f(VariableBuildInfo(sbi.mesh(),"SplitDeltaT"))
, m_old_dt_f(VariableBuildInfo(sbi.mesh(),"OldDTf"))
, m_module(nullptr)
{
  if (TaskFactory::isActive()){
    info() << "USE CONCURRENCY!!!!!";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHydroSimdService::
~SimpleHydroSimdService()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHydroSimdService::
hydroBuild()
{
  info() << "Using hydro with vectorisation name=" << SimdInfo::name()
         << " vector_size=" << SimdReal::Length << " index_size=" << SimdInfo::Int32IndexSize;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHydroSimdService::
hydroExit()
{
  info() << "Hydro exit entry point";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialisation du module hydro lors du démarrage du cas.
 */
void SimpleHydroSimdService::
hydroStartInit()
{
  info() << "START_INIT sizeof(ItemLocalId)=" << sizeof(ItemLocalId);
  ENUMERATE_CELL(icell,allCells()){
    m_sub_domain_id[icell]=subDomain()->subDomainId();
    m_cell_unique_id[icell]=icell->uniqueId();
  }

  // Dimensionne les variables tableaux
  m_cell_cqs.resize(8);
  
  //info() << "SimpleHydro SIMD initialisation vec_size=" << SimdReal::BLOCK_SIZE;

  // Vérifie que les valeurs initiales sont correctes
  {
    Integer nb_error = 0;
    auto in_pressure = viewIn(m_pressure);
    auto in_adiabatic_cst = viewIn(m_adiabatic_cst);
    VariableCellRealInView in_density = viewIn(m_density);
    ENUMERATE_CELL(icell,allCells()){
      CellLocalId cid { icell.asItemLocalId() };
      Real pressure = in_pressure[cid];
      Real adiabatic_cst = in_adiabatic_cst[cid];
      Real density = in_density[cid];
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
  Real deltat_init = m_module->getDeltatInit();
  m_delta_t_n = deltat_init;
  m_delta_t_f = deltat_init;

  // Initialise les données géométriques: volume, cqs, longueurs caractéristiques
  computeGeometricValues();

  m_node_mass.fill(ARCANE_REAL(0.0));
  m_velocity.fill(Real3::zero());

  // Initialisation de la masses des mailles et des masses nodale
  ENUMERATE_CELL(icell,allCells()){
    Cell cell = *icell;
    m_cell_mass[icell] = m_density[icell] * m_volume[icell];

    Real contrib_node_mass = ARCANE_REAL(0.125) * m_cell_mass[cell];
    for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node ){
      m_node_mass[i_node] += contrib_node_mass;
    }
  }

  m_node_mass.synchronize();

  // Initialise l'énergie et la vitesse du son
  auto in_pressure = viewIn(m_pressure);
  auto in_density = viewIn(m_density);
  auto in_adiabatic_cst = viewIn(m_adiabatic_cst);

  VariableCellRealOutView out_internal_energy = viewOut(m_internal_energy);
  auto out_sound_speed = viewOut(m_sound_speed);

  ENUMERATE_SIMD_CELL(icell,allCells()){
    SimdCell vi = *icell;
    SimdReal pressure = in_pressure[vi];
    SimdReal adiabatic_cst = in_adiabatic_cst[vi];
    SimdReal density = in_density[vi];
    out_internal_energy[vi] = pressure / ((adiabatic_cst-ARCANE_REAL(1.0)) * density);
    out_sound_speed[vi] = math::sqrt(adiabatic_cst*pressure/density);
  }
  info() << "END_START_INIT";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des forces au temps courant \f$t^{n}\f$
 */
void SimpleHydroSimdService::
computeForces()
{
  // Remise à zéro du vecteur des forces.
  m_force.fill(Real3::null());

  VariableNodeReal3OutView out_force = viewOut(m_force);
  VariableNodeReal3InView in_force = viewIn(m_force);

  // Calcul pour chaque noeud de chaque maille la contribution
  // des forces de pression et de la pseudo-viscosite si necessaire
  if (m_module->getViscosity()==TypesSimpleHydro::ViscosityCellScalar){
    _computePressureAndCellPseudoViscosityForces();
  }
  else{
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Real pressure = m_pressure[cell];
      for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node ){
        NodeLocalId nid = *i_node;
        out_force[nid] = in_force[nid] + pressure * m_cell_cqs[icell][i_node.index()];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Pseudo viscosité scalaire aux mailles
 */
void SimpleHydroSimdService::
_computePressureAndCellPseudoViscosityForces()
{
  Real linear_coef = m_module->getViscosityLinearCoef();
  Real quadratic_coef = m_module->getViscosityQuadraticCoef();

  auto in_pressure = viewIn(m_pressure);
  auto in_density = viewIn(m_density);
  auto out_cell_viscosity_force = viewOut(m_cell_viscosity_force);

  // Boucle sur les mailles du maillage
  ENUMERATE_CELL(icell,allCells().view()){
    Cell cell = *icell;
    CellLocalId cid(cell);

    const Real rho = in_density[cell];
    const Real pressure = in_pressure[icell];

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
      out_cell_viscosity_force[cid] = scalar_viscosity;
      for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node )
        m_force[i_node] += (pressure + scalar_viscosity)*m_cell_cqs[icell][i_node.index()];
    }
    else{
      out_cell_viscosity_force[cid] = ARCANE_REAL(0.0);
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
void SimpleHydroSimdService::
computeVelocity()
{
  m_force.synchronize();

  auto in_old_velocity = viewIn(m_velocity);
  auto in_node_mass = viewIn(m_node_mass);
  auto in_force = viewIn(m_force);
  auto out_velocity = viewOut(m_velocity);
  // Calcule l'impulsion aux noeuds
  ENUMERATE_SIMD_NODE(i_node,allNodes()){
    SimdNode node = *i_node;
    SimdReal node_mass  = in_node_mass[node];
    SimdReal3 old_velocity = in_old_velocity[node];
    SimdReal3 new_velocity = old_velocity + (m_delta_t_n() / node_mass) * in_force[i_node];
    out_velocity[node] = new_velocity;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase3).
 */
void SimpleHydroSimdService::
computeViscosityWork()
{
  arcaneParallelForeach(allCells(),[this](CellVectorView cells){

      auto in_cell_viscosity_force = viewIn(m_cell_viscosity_force);
      auto out_viscosity_work = viewOut(m_viscosity_work);

      // Calcul du travail des forces de viscosité dans une maille
      ENUMERATE_CELL(icell,cells){
        Cell cell = *icell; 
        CellLocalId cid(cell);
        Real work = 0.;
        Real scalar_viscosity = in_cell_viscosity_force[cid];
        if (!math::isZero(scalar_viscosity))
          for( NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node )
            work += math::scaMul(scalar_viscosity*m_cell_cqs[icell][i_node.index()],m_velocity[i_node]);
        out_viscosity_work[cid] = work;
      }
    }
    );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prise en compte des conditions aux limites.
 */
void SimpleHydroSimdService::
applyBoundaryCondition()
{
  for( auto bc : m_module->getBoundaryConditions() ){
    FaceGroup face_group = bc->getSurface();
    NodeGroup node_group = face_group.nodeGroup();
    Real value = bc->getValue();
    TypesSimpleHydro::eBoundaryCondition type = bc->getType();

    // boucle sur les faces de la surface
    ENUMERATE_FACE(j,face_group){
      Face face = *j;
      Integer nb_node = face.nbNode();

      // boucle sur les noeuds de la face
      for( Integer k=0; k<nb_node; ++k ){
        Node node = face.node(k);
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
void SimpleHydroSimdService::
moveNodes()
{
  Real deltat_f = m_delta_t_f();
  
  ENUMERATE_NODE(i_node,allNodes()){
    m_node_coord[i_node] += deltat_f * m_velocity[i_node];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mise à jour des densités et calcul de l'accroissements max
 *	  de la densité sur l'ensemble du maillage.
 */
void SimpleHydroSimdService::
updateDensity()
{
  Real density_ratio_maximum = ARCANE_REAL(0.0);
  
  HYDRO_PRAGMA_IVDEP
  ENUMERATE_CELL(icell,allCells()){
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
void SimpleHydroSimdService::
applyEquationOfState()
{
  const Real deltatf = m_delta_t_f();
  const bool add_viscosity_force = (m_module->getViscosity()!=TypesSimpleHydro::ViscosityNo);
  
  auto in_adiabatic_cst = viewIn(m_adiabatic_cst);
  auto in_volume = viewIn(m_volume);
  auto in_density = viewIn(m_density);
  auto in_old_volume = viewIn(m_old_volume);
  auto in_internal_energy = viewIn(m_internal_energy);
  auto in_cell_mass = viewIn(m_cell_mass);
  auto in_viscosity_work = viewIn(m_viscosity_work);

  auto out_internal_energy = viewOut(m_internal_energy);
  auto out_sound_speed = viewOut(m_sound_speed);
  auto out_pressure = viewOut(m_pressure);

  // Calcul de l'énergie interne
  arcaneParallelForeach(allCells(),[&](CellVectorView cells){
      ENUMERATE_SIMD_CELL(icell,cells){
        SimdCell vi = *icell;
        SimdReal adiabatic_cst = in_adiabatic_cst[vi];
        SimdReal volume_ratio = in_volume[vi] / in_old_volume[vi];
        SimdReal x = ARCANE_REAL(0.5)*(adiabatic_cst-ARCANE_REAL(1.0));
        SimdReal numer_accrois_nrj = ARCANE_REAL(1.0) + x*(ARCANE_REAL(1.0)-volume_ratio);
        SimdReal denom_accrois_nrj = ARCANE_REAL(1.0) + x*(ARCANE_REAL(1.0)-(ARCANE_REAL(1.0)/volume_ratio));
        SimdReal internal_energy = in_internal_energy[vi];
        internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);

        // Prise en compte du travail des forces de viscosité 
        if (add_viscosity_force)
          internal_energy = internal_energy - deltatf*in_viscosity_work[vi] / (in_cell_mass[vi]*denom_accrois_nrj);

        out_internal_energy[vi] = internal_energy;

        SimdReal density = in_density[vi];
        SimdReal pressure = (adiabatic_cst-ARCANE_REAL(1.0)) * density * internal_energy; 
        SimdReal sound_speed = math::sqrt(adiabatic_cst*pressure/density);
        out_pressure[vi] = pressure;
        out_sound_speed[vi] = sound_speed;
      }
    }
    );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des nouveaux pas de temps.
 */
void SimpleHydroSimdService::
computeDeltaT()
{
  const Real old_dt = m_global_deltat();

  // Calcul du pas de temps pour le respect du critère de CFL
  
  Real minimum_aux = FloatInfo<Real>::maxValue();
  Real new_dt = FloatInfo<Real>::maxValue();

  HYDRO_PRAGMA_IVDEP
  ENUMERATE_CELL(icell,ownCells()){
    Real cell_dx = m_caracteristic_length[icell];
    Real sound_speed = m_sound_speed[icell];
    Real dx_sound = cell_dx / sound_speed;
    minimum_aux = math::min(minimum_aux,dx_sound);
  }

  new_dt = m_module->getCfl()*minimum_aux;

  // Pas de variations trop brutales à la hausse comme à la baisse
  Real max_dt = (ARCANE_REAL(1.0)+m_module->getVariationSup())*old_dt;
  Real min_dt = (ARCANE_REAL(1.0)-m_module->getVariationInf())*old_dt;

  new_dt = math::min(new_dt,max_dt);
  new_dt = math::max(new_dt,min_dt);

  // control de l'accroissement relatif de la densité
  Real dgr = m_module->getDensityGlobalRatio();
  if (m_density_ratio_maximum()>dgr)
    new_dt = math::min(old_dt*dgr/m_density_ratio_maximum(),new_dt);

  IParallelMng* pm = mesh()->parallelMng();
  new_dt = pm->reduce(Parallel::ReduceMin,new_dt);

  // Respect des valeurs min et max imposées par le fichier de données .plt
  new_dt = math::min(new_dt,m_module->getDeltatMax());
  new_dt = math::max(new_dt,m_module->getDeltatMin());

  // Le dernier calcul se fait exactement au temps stopTime()
  {
    Real stop_time  = m_module->getFinalTime();
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
inline void SimpleHydroSimdService::
computeCQsSimd(SimdReal3 node_coord[8],SimdReal3 face_coord[6],SimdReal3 cqs[8])
{
  const SimdReal3 c0 = face_coord[0];
  const SimdReal3 c1 = face_coord[1];
  const SimdReal3 c2 = face_coord[2];
  const SimdReal3 c3 = face_coord[3];
  const SimdReal3 c4 = face_coord[4];
  const SimdReal3 c5 = face_coord[5];

  const Real demi = ARCANE_REAL(0.5);
  const Real five = ARCANE_REAL(5.0);

  // Calcul des normales face 1 :
  const SimdReal3 n1a04 = demi * math::cross(node_coord[0] - c0 , node_coord[3] - c0);
  const SimdReal3 n1a03 = demi * math::cross(node_coord[3] - c0 , node_coord[2] - c0);
  const SimdReal3 n1a02 = demi * math::cross(node_coord[2] - c0 , node_coord[1] - c0);
  const SimdReal3 n1a01 = demi * math::cross(node_coord[1] - c0 , node_coord[0] - c0);

  // Calcul des normales face 2 :
  const SimdReal3 n2a05 = demi * math::cross(node_coord[0] - c1 , node_coord[4] - c1);
  const SimdReal3 n2a12 = demi * math::cross(node_coord[4] - c1 , node_coord[7] - c1);
  const SimdReal3 n2a08 = demi * math::cross(node_coord[7] - c1 , node_coord[3] - c1);
  const SimdReal3 n2a04 = demi * math::cross(node_coord[3] - c1 , node_coord[0] - c1);

  // Calcul des normales face 3 :
  const SimdReal3 n3a01 = demi * math::cross(node_coord[0] - c2 , node_coord[1] - c2);
  const SimdReal3 n3a06 = demi * math::cross(node_coord[1] - c2 , node_coord[5] - c2);
  const SimdReal3 n3a09 = demi * math::cross(node_coord[5] - c2 , node_coord[4] - c2);
  const SimdReal3 n3a05 = demi * math::cross(node_coord[4] - c2 , node_coord[0] - c2);

  // Calcul des normales face 4 :
  const SimdReal3 n4a09 = demi * math::cross(node_coord[4] - c3 , node_coord[5] - c3);
  const SimdReal3 n4a10 = demi * math::cross(node_coord[5] - c3 , node_coord[6] - c3);
  const SimdReal3 n4a11 = demi * math::cross(node_coord[6] - c3 , node_coord[7] - c3);
  const SimdReal3 n4a12 = demi * math::cross(node_coord[7] - c3 , node_coord[4] - c3);
	
  // Calcul des normales face 5 :
  const SimdReal3 n5a02 = demi * math::cross(node_coord[1] - c4 , node_coord[2] - c4);
  const SimdReal3 n5a07 = demi * math::cross(node_coord[2] - c4 , node_coord[6] - c4);
  const SimdReal3 n5a10 = demi * math::cross(node_coord[6] - c4 , node_coord[5] - c4);
  const SimdReal3 n5a06 = demi * math::cross(node_coord[5] - c4 , node_coord[1] - c4);
      
  // Calcul des normales face 6 :
  const SimdReal3 n6a03 = demi * math::cross(node_coord[2] - c5 , node_coord[3] - c5);
  const SimdReal3 n6a08 = demi * math::cross(node_coord[3] - c5 , node_coord[7] - c5);
  const SimdReal3 n6a11 = demi * math::cross(node_coord[7] - c5 , node_coord[6] - c5);
  const SimdReal3 n6a07 = demi * math::cross(node_coord[6] - c5 , node_coord[2] - c5);

  const Real real_1div12 = ARCANE_REAL(1.0) / ARCANE_REAL(12.0);

  cqs[0] = (five*(n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
            (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09))*real_1div12;
  cqs[1] = (five*(n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
            (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07))*real_1div12;
  cqs[2] = (five*(n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
            (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08))*real_1div12;
  cqs[3] = (five*(n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
            (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11))*real_1div12;
  cqs[4] = (five*(n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
            (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11))*real_1div12;
  cqs[5] = (five*(n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
            (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02))*real_1div12;
  cqs[6] = (five*(n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
            (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08))*real_1div12;
  cqs[7] = (five*(n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
            (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03))*real_1div12;

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du volume des mailles, des longueurs caractéristiques
 * et des résultantes aux sommets.
 */
void SimpleHydroSimdService::
computeGeometricValues()
{
  auto out_caracteristic_length = viewOut(m_caracteristic_length);
  arcaneParallelForeach(allCells(),[&](CellVectorView cells){
      // Copie locale des coordonnées des sommets d'une maille
      SimdReal3 coord[8];
      // Coordonnées des centres des faces
      SimdReal3 face_coord[6];

      SimdReal3 cqs[8];
      //std::cerr << "SIZE=" << cells.size() << '\n';
      ENUMERATE_SIMD_CELL(ivecitem,cells){
        SimdCell vitem = *ivecitem;
        for( CellEnumerator iscell(ivecitem.enumerator()); iscell.hasNext(); ++iscell ){
          Cell cell(*iscell);
          Integer si(iscell.index());
          // Recopie les coordonnées locales (pour le cache)
          for( NodeEnumerator i_node(cell.nodes()); i_node.index()<8; ++i_node ){
            coord[i_node.index()].set(si,m_node_coord[i_node]);
            //info() << "COORD lid=" << cell.localId() << " i=" << i << " index=" << i_node.index() << " v=" << coord[i_node.index()][i];
            }
        }

        // Calcul les coordonnées des centres des faces
        face_coord[0] = ARCANE_REAL(0.25) * ( coord[0] + coord[3] + coord[2] + coord[1] );
        face_coord[1] = ARCANE_REAL(0.25) * ( coord[0] + coord[4] + coord[7] + coord[3] );
        face_coord[2] = ARCANE_REAL(0.25) * ( coord[0] + coord[1] + coord[5] + coord[4] );
        face_coord[3] = ARCANE_REAL(0.25) * ( coord[4] + coord[5] + coord[6] + coord[7] );
        face_coord[4] = ARCANE_REAL(0.25) * ( coord[1] + coord[2] + coord[6] + coord[5] );
        face_coord[5] = ARCANE_REAL(0.25) * ( coord[2] + coord[3] + coord[7] + coord[6] );

        // Calcule la longueur caractéristique de la maille.
        SimdReal3 median1 = face_coord[0]-face_coord[3];
        SimdReal3 median2 = face_coord[2]-face_coord[5];
        SimdReal3 median3 = face_coord[1]-face_coord[4];
        SimdReal d1 = math::normL2(median1);
        SimdReal d2 = math::normL2(median2);
        SimdReal d3 = math::normL2(median3);
          
        SimdReal dx_numerator   = d1*d2*d3;
        SimdReal dx_denominator = d1*d2 + d1*d3 + d2*d3;
        out_caracteristic_length[vitem] = dx_numerator / dx_denominator;

        //for( Integer i=0; i<NV; ++i ){
          //Cell cell(vitem.item(i));
          // Calcule les résultantes aux sommets
        computeCQsSimd(coord,face_coord,cqs);
        //}
        // Calcul des résultantes aux sommets :
        for( CellEnumerator si(ivecitem.enumerator()); si.hasNext(); ++si ){
          Cell cell(*si);
          Integer sidx(si.index());
          ArrayView<Real3> cqsv = m_cell_cqs[cell];
          for( Integer i_node=0; i_node<8; ++i_node )
            cqsv[i_node] = cqs[i_node][sidx];
        }
        
        /*for( Integer i=0; i<NV; ++i ){
          Cell cell(vitem.item(i));
          for( Integer z=0; z<8; ++z ){
            info() << "CQS lid=" << cell.localId() << " i=" << i << " z=" << z << " v=" << m_cell_cqs[cell][z];
          }
          }*/

        // Calcule le volume de la maille
        for( CellEnumerator si(ivecitem.enumerator()); si.hasNext(); ++si ){
          Cell cell(*si);
          Integer sidx(si.index());
          Real volume = 0.;
          for( Integer i_node=0; i_node<8; ++i_node )
            volume += math::dot(coord[i_node][sidx],cqs[i_node][sidx]);
          volume /= ARCANE_REAL(3.0);
          
          m_old_volume[cell] = m_volume[cell];
          m_volume[cell] = volume;
        }
        /*SimdReal volume(0.0);
        for( Integer i_node=0; i_node<8; ++i_node )
          volume = volume + math::dot(coord[i_node],cqs[i_node]);
        volume = volume / ARCANE_REAL(3.0);

        for( Integer i=0; i<NV; ++i ){
          Cell cell(vitem.item(i));
          m_old_volume[cell] = m_volume[cell];
          m_volume[cell] = volume[i];
          }*/
      }
    }
);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHydroSimdService::
hydroInit()
{
  if (m_module->getBackwardIteration()!=0){
    subDomain()->timeLoopMng()->setBackwardSavePeriod(10);
  }
  info() << "INIT: DTmin=" << m_module->getDeltatMin()
         << " DTmax="<< m_module->getDeltatMax()
         << " DT="<< m_global_deltat();
  if (m_global_deltat() > m_module->getDeltatMax())
    ARCANE_FATAL("DeltaT > DTMax");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SimpleHydroSimdService,
                        ServiceProperty("StdHydroSimdService",ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(SimpleHydro::ISimpleHydroService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace SimpleHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
