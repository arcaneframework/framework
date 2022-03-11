// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MicroHydroModule.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Bench MicroHydro.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/MathUtils.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/VariableTypes.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IParallelMng.h"
#include "arcane/ModuleFactory.h"
#include "arcane/ItemPrinter.h"
#include "arcane/ITimeStats.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/mesh/ItemFamily.h"

#include "arcane/UnstructuredMeshConnectivity.h"

#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/RunCommandEnumerate.h"

#include "MicroHydroTypes.h"
#include "MicroHydro_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MicroHydro
{

namespace ax = Arcane::Accelerator;
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Valeurs des références pour la validation

double reference_density_ratio_maximum[50] = {
  0.0160000000000123, 0.00170124869728498, 0.00204753556227649, 0.00246464898229085, 0.00296699303969296,
  0.00357184605321602, 0.00429990753718348, 0.00517593569890155, 0.00622948244180328, 0.00749572737143143,
  0.0090164010818345, 0.0108407663681081, 0.0130265854869945, 0.0156409265893111, 0.0187605241309118,
  0.0224711528439701, 0.0268650034204973, 0.0320341788827704, 0.0380568426206173, 0.0449697158584332,
  0.0527158336840399, 0.0610493482313262, 0.0693720874578793, 0.0764829000430237, 0.0802862351795616,
  0.0670829131533573, 0.0597808473490852, 0.0654293866188612, 0.0698233985169538, 0.0724886365903765,
  0.0728380430958174, 0.0702035766339136, 0.0641244580818233, 0.0547358261067183, 0.0568567736303723,
  0.06387075263142, 0.0694149111684302, 0.0726551742001545, 0.072829643927776, 0.0693970119163948,
  0.0622148000416855, 0.0517264119336657, 0.0592227788170703, 0.0665696529262891, 0.0717611469166364,
  0.0741457462067702, 0.0730964375784472, 0.0682069575016942, 0.0596677042759776, 0.0543168109309979
};

double reference_global_deltat[50] = {
  0.000000000000000e+00, 1.000000000000000e-04, 1.100000000000000e-04, 1.210000000000000e-04, 1.331000000000000e-04,
  1.464100000000001e-04, 1.610510000000001e-04, 1.771561000000001e-04, 1.948717100000001e-04, 2.143588810000001e-04,
  2.357947691000002e-04, 2.593742460100002e-04, 2.853116706110002e-04, 3.138428376721002e-04, 3.452271214393103e-04,
  3.797498335832414e-04, 4.177248169415655e-04, 4.594972986357221e-04, 5.054470284992944e-04, 5.559917313492239e-04,
  6.115909044841464e-04, 6.727499949325611e-04, 7.400249944258172e-04, 8.140274938683990e-04, 8.954302432552390e-04,
  8.342344802558105e-04, 7.832988697246865e-04, 7.455699714960305e-04, 7.189393815728784e-04, 7.011461654789450e-04,
  6.900170474392408e-04, 6.835626230429761e-04, 6.800416216153530e-04, 6.780401694886634e-04, 6.765567003929260e-04,
  6.750478953733179e-04, 6.733949061097245e-04, 6.717890229153485e-04, 6.705765765568119e-04, 6.701124635250109e-04,
  6.706540214144399e-04, 6.723370623794280e-04, 6.751721250357102e-04, 6.790835285303133e-04, 6.783102389627785e-04,
  6.765647969871316e-04, 6.753134577141526e-04, 6.751179527299745e-04, 6.762828720782492e-04, 6.789494792300497e-04
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module MicroHydro.
 *
 * Ce module implémente une hydrodynamique simple tri-dimensionnel,
 * parallèle, avec une pseudo-viscosité aux mailles.
 */
class MicroHydroModule
: public ArcaneMicroHydroObject
{
 public:

  struct BoundaryCondition
  {
    NodeGroup nodes;
    NodeVectorView view;
    Real value;
    MicroHydroTypes::eBoundaryCondition type;
  };

  // Note: il faut mettre ce champs statique si on veut que sa valeur
  // soit correcte lors de la capture avec CUDA (sinon on passe par this et
  // cela provoque une erreur mémoire)
  static const Integer MAX_NODE_CELL = 8;

 public:

  //! Constructeur
  explicit MicroHydroModule(const ModuleBuildInfo& mb);

 public:

  VersionInfo versionInfo() const override { return VersionInfo(2, 0, 1); }

 public:

  void hydroBuild() override;
  void hydroStartInit() override;
  void hydroInit() override;
  void hydroExit() override;
  void hydroOnMeshChanged() override;
  void doOneIteration() override;

 public:

  void computeForces();
  void computeVelocity();
  void computeViscosityWork();
  void applyBoundaryCondition();
  void moveNodes();
  void computeGeometricValues();
  void updateDensity();
  void applyEquationOfState();
  void computeDeltaT();

 private:

  ITimeStats* m_time_stats = nullptr;
  Timer m_elapsed_timer;

  //! Indice de chaque noeud dans la maille
  UniqueArray<Int16> m_node_index_in_cells;

  ax::Runner* m_runner = nullptr;

  UnstructuredMeshConnectivityView m_connectivity_view;
  UniqueArray<BoundaryCondition> m_boundary_conditions;

 private:

  void _computePressureAndCellPseudoViscosityForces();

 private:

  void _specialInit();
  void _computeNodeIndexInCells();
  void _doCall(const char* func_name, std::function<void()> func);
  void computeGeometricValues2();

  void cellScalarPseudoViscosity();
  ARCCORE_HOST_DEVICE inline void computeCQs(Real3 node_coord[8], Real3 face_coord[6], Span<Real3> cqs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MicroHydroModule::
MicroHydroModule(const ModuleBuildInfo& sbi)
: ArcaneMicroHydroObject(sbi)
, m_time_stats(sbi.subDomain()->timeStats())
, m_elapsed_timer(sbi.subDomain(), "MicroHydro", Timer::TimerReal)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
hydroBuild()
{
  info() << "Bench MicroHydro";
  m_runner = acceleratorMng()->defaultRunner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialisation du module hydro lors du démarrage du cas.
 */
void MicroHydroModule::
hydroStartInit()
{
  m_connectivity_view.setMesh(this->mesh());

  // Dimensionne les variables tableaux
  m_cell_cqs.resize(8);
  _computeNodeIndexInCells();

  // Vérifie que les valeurs initiales sont correctes
  {
    Integer nb_error = 0;
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto in_pressure = ax::viewIn(command, m_pressure);
    auto in_adiabatic_cst = ax::viewIn(command, m_adiabatic_cst);
    ax::VariableCellRealInView in_density = ax::viewIn(command, m_density);
    ENUMERATE_CELL (icell, allCells()) {
      CellLocalId cid = *icell;
      Real pressure = in_pressure[cid];
      Real adiabatic_cst = in_adiabatic_cst[cid];
      Real density = in_density[cid];
      if (math::isZero(pressure) || math::isZero(density) || math::isZero(adiabatic_cst)) {
        info() << "Null valeur for cell=" << ItemPrinter(*icell)
               << " density=" << density
               << " pressure=" << pressure
               << " adiabatic_cst=" << adiabatic_cst;
        ++nb_error;
      }
    }
    if (nb_error != 0)
      ARCANE_FATAL("Some ({0}) cells are not initialised", nb_error);
  }

  // Initialise le delta-t
  Real deltat_init = options()->getDeltatInit();
  m_delta_t_n = deltat_init;
  m_delta_t_f = deltat_init;

  // Initialise les données géométriques: volume, cqs, longueurs caractéristiques
  computeGeometricValues();

  m_node_mass.fill(ARCANE_REAL(0.0));
  m_velocity.fill(Real3::zero());

  // Initialisation de la masses des mailles et des masses nodale
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    m_cell_mass[icell] = m_density[icell] * m_volume[icell];

    Real contrib_node_mass = ARCANE_REAL(0.125) * m_cell_mass[cell];
    for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node) {
      m_node_mass[i_node] += contrib_node_mass;
    }
  }

  m_node_mass.synchronize();

  {
    info() << "Initialize SoundSpeed and InternalEnergy";
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    // Initialise l'énergie et la vitesse du son
    auto in_pressure = ax::viewIn(command, m_pressure);
    auto in_density = ax::viewIn(command, m_density);
    auto in_adiabatic_cst = ax::viewIn(command, m_adiabatic_cst);

    auto out_internal_energy = ax::viewOut(command, m_internal_energy);
    auto out_sound_speed = ax::viewOut(command, m_sound_speed);

    command << RUNCOMMAND_ENUMERATE(Cell, vi, allCells())
    {
      Real pressure = in_pressure[vi];
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real density = in_density[vi];
      out_internal_energy[vi] = pressure / ((adiabatic_cst - 1.0) * density);
      out_sound_speed[vi] = math::sqrt(adiabatic_cst * pressure / density);
    };
  }

  // Remplit la structure contenant les informations sur les conditions aux limites
  // Cela permet de garantir avec les accélérateurs qu'on pourra accéder
  // de manière concurrente aux données.
  {
    m_boundary_conditions.clear();
    for (auto bc : options()->getBoundaryCondition()) {
      FaceGroup face_group = bc->getSurface();
      Real value = bc->getValue();
      MicroHydroTypes::eBoundaryCondition type = bc->getType();
      BoundaryCondition bcn;
      bcn.nodes = face_group.nodeGroup();
      bcn.value = value;
      bcn.type = type;
      m_boundary_conditions.add(bcn);
    }
  }
  info() << "END_START_INIT";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Point d'entrée appelé après un équilibrage de charge.
 *
 * Il faut reconstruire les informations de connectivités propres à notre module.
 */
void MicroHydroModule::
hydroOnMeshChanged()
{
  info() << "Hydro: OnMeshChanged";

  m_connectivity_view.setMesh(this->mesh());
  _computeNodeIndexInCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des forces au temps courant \f$t^{n}\f$
 */
void MicroHydroModule::
computeForces()
{
  // Calcul pour chaque noeud de chaque maille la contribution
  // des forces de pression et de la pseudo-viscosite si necessaire

  Real linear_coef = options()->getViscosityLinearCoef();
  Real quadratic_coef = options()->getViscosityQuadraticCoef();

  auto cnc = m_connectivity_view.cellNode();

  // Calcul de la divergence de la vitesse et de la viscosité scalaire
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto in_density = viewIn(command, m_density);
    auto in_velocity = viewIn(command, m_velocity);
    auto in_caracteristic_length = viewIn(command, m_caracteristic_length);
    auto in_volume = viewIn(command, m_volume);
    auto in_sound_speed = viewIn(command, m_sound_speed);
    auto in_cell_cqs = viewIn(command, m_cell_cqs);
    auto out_cell_viscosity_force = viewOut(command, m_cell_viscosity_force);
    command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells())
    {
      Real delta_speed = 0.0;
      Int32 i = 0;
      for (NodeLocalId node : cnc.nodes(cid)) {
        delta_speed += math::dot(in_velocity[node], in_cell_cqs[cid][i]);
        ++i;
      }
      delta_speed /= in_volume[cid];

      // Capture uniquement les chocs
      bool shock = (math::min(ARCANE_REAL(0.0), delta_speed) < ARCANE_REAL(0.0));
      if (shock) {
        Real rho = in_density[cid];
        Real sound_speed = in_sound_speed[cid];
        Real dx = in_caracteristic_length[cid];
        Real quadratic_viscosity = rho * dx * dx * delta_speed * delta_speed;
        Real linear_viscosity = -rho * sound_speed * dx * delta_speed;
        Real scalar_viscosity = linear_coef * linear_viscosity + quadratic_coef * quadratic_viscosity;
        out_cell_viscosity_force[cid] = scalar_viscosity;
      }
      else {
        out_cell_viscosity_force[cid] = 0.0;
      }
    };
  }

  constexpr int max_node_cell = MAX_NODE_CELL;
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto in_pressure = viewIn(command, m_pressure);
    auto in_cell_viscosity_force = viewIn(command, m_cell_viscosity_force);
    auto in_cell_cqs = viewIn(command, m_cell_cqs);
    auto out_force = viewOut(command, m_force);
    auto node_index_in_cells = m_node_index_in_cells.constSpan();
    auto nc_cty = m_connectivity_view.nodeCell();
    command << RUNCOMMAND_ENUMERATE(Node, node, allNodes())
    {
      Int32 first_pos = node.localId() * max_node_cell;
      Real3 force;
      Integer index = 0;
      for (CellLocalId cell : nc_cty.cells(node)) {
        Int16 node_index = node_index_in_cells[first_pos + index];
        Real scalar_viscosity = in_cell_viscosity_force[cell];
        Real pressure = in_pressure[cell];
        force += (pressure + scalar_viscosity) * in_cell_cqs[cell][node_index];
        ++index;
      }
      out_force[node] = force;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase2).
 */
void MicroHydroModule::
computeVelocity()
{
  m_force.synchronize();

  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  auto in_node_mass = viewIn(command, m_node_mass);
  auto in_force = viewIn(command, m_force);
  auto in_out_velocity = viewInOut(command, m_velocity);
  Real delta_t_n = m_delta_t_n();

  // Calcule l'impulsion aux noeuds
  command << RUNCOMMAND_ENUMERATE(Node, node, allNodes())
  {
    Real node_mass = in_node_mass[node];
    Real3 old_velocity = in_out_velocity[node];
    Real3 new_velocity = old_velocity + (delta_t_n / node_mass) * in_force[node];
    in_out_velocity[node] = new_velocity;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase3).
 */
void MicroHydroModule::
computeViscosityWork()
{
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  auto in_cell_viscosity_force = viewIn(command, m_cell_viscosity_force);
  auto in_velocity = viewIn(command, m_velocity);
  auto out_viscosity_work = viewOut(command, m_viscosity_work);
  auto in_cell_cqs = viewIn(command, m_cell_cqs);
  auto cnc = m_connectivity_view.cellNode();

  // Calcul du travail des forces de viscosité dans une maille
  command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells())
  {
    Real work = 0.0;
    Real scalar_viscosity = in_cell_viscosity_force[cid];
    if (!math::isZero(scalar_viscosity)) {
      Integer i = 0;
      for (NodeLocalId node : cnc.nodes(cid)) {
        work += math::dot(scalar_viscosity * in_cell_cqs[cid][i], in_velocity[node]);
        ++i;
      }
    }
    out_viscosity_work[cid] = work;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prise en compte des conditions aux limites.
 */
void MicroHydroModule::
applyBoundaryCondition()
{
  auto queue = makeQueue(m_runner);

  // Pour cette méthode, comme les conditions aux limites sont sur des groupes
  // indépendants (ou alors avec la même valeur si c'est sur les mêmes noeuds),
  // on peut exécuter les noyaux en asynchrone.
  queue.setAsync(true);

  // Repositionne les vues si les groupes associés ont été modifiés
  for (auto& bc : m_boundary_conditions)
    bc.view = bc.nodes.view();
  for (auto bc : m_boundary_conditions) {
    Real value = bc.value;
    MicroHydroTypes::eBoundaryCondition type = bc.type;
    NodeVectorView view = bc.view;

    auto command = makeCommand(queue);
    auto in_out_velocity = viewInOut(command, m_velocity);
    // boucle sur les faces de la surface
    command << RUNCOMMAND_ENUMERATE(Node, node, view)
    {
      // boucle sur les noeuds de la face
      Real3 v = in_out_velocity[node];
      switch (type) {
      case MicroHydroTypes::VelocityX:
        v.x = value;
        break;
      case MicroHydroTypes::VelocityY:
        v.y = value;
        break;
      case MicroHydroTypes::VelocityZ:
        v.z = value;
        break;
      case MicroHydroTypes::Unknown:
        break;
      }
      in_out_velocity[node] = v;
    };
  }
  queue.barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Déplace les noeuds.
 */
void MicroHydroModule::
moveNodes()
{
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  Real deltat_f = m_delta_t_f();

  auto in_velocity = viewIn(command, m_velocity);
  auto in_out_node_coord = viewInOut(command, m_node_coord);

  command << RUNCOMMAND_ENUMERATE(Node, node, allNodes())
  {
    Real3 coord = in_out_node_coord[node];
    in_out_node_coord[node] = coord + (deltat_f * in_velocity[node]);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mise à jour des densités et calcul de l'accroissements max
 *	  de la densité sur l'ensemble du maillage.
 */
void MicroHydroModule::
updateDensity()
{
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  ax::ReducerMax<double> density_ratio_maximum(command);
  density_ratio_maximum.setValue(0.0);
  auto in_cell_mass = viewIn(command, m_cell_mass);
  auto in_volume = viewIn(command, m_volume);
  auto in_out_density = viewInOut(command, m_density);

  command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells())
  {
    Real old_density = in_out_density[cid];
    Real new_density = in_cell_mass[cid] / in_volume[cid];

    in_out_density[cid] = new_density;

    Real density_ratio = (new_density - old_density) / new_density;

    density_ratio_maximum.max(density_ratio);
  };

  m_density_ratio_maximum = density_ratio_maximum.reduce();

  // Vérifie la validité du ratio calculé. La référence n'est valide
  // qu'en séquentiel car ce ratio n'est pas réduit sur tout les
  // sous-domaines.
  if (options()->getCheckNumericalResult()) {
    if (!mesh()->parallelMng()->isParallel()) {
      Integer iteration = m_global_iteration();
      if (iteration <= 50) {
        Real max_dr = m_density_ratio_maximum();
        Real ref_max_dr = reference_density_ratio_maximum[iteration - 1];
        if (!math::isNearlyEqualWithEpsilon(max_dr, ref_max_dr, 1.0e-12))
          ARCANE_FATAL("Bad value for density_ratio_maximum: ref={0} v={1} diff={2}",
                       ref_max_dr, max_dr, (ref_max_dr - max_dr) / ref_max_dr);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'équation d'état et calcul l'énergie interne et la
 * pression.
 */
void MicroHydroModule::
applyEquationOfState()
{
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  const Real deltatf = m_delta_t_f();
  const bool add_viscosity_force = true;

  auto in_adiabatic_cst = viewIn(command, m_adiabatic_cst);
  auto in_volume = viewIn(command, m_volume);
  auto in_density = viewIn(command, m_density);
  auto in_old_volume = viewIn(command, m_old_volume);
  auto in_cell_mass = viewIn(command, m_cell_mass);
  auto in_viscosity_work = viewIn(command, m_viscosity_work);

  auto in_out_internal_energy = viewInOut(command, m_internal_energy);
  auto out_sound_speed = viewOut(command, m_sound_speed);
  auto out_pressure = viewOut(command, m_pressure);

  // Calcul de l'énergie interne
  command << RUNCOMMAND_ENUMERATE(Cell, vi, allCells())
  {
    Real adiabatic_cst = in_adiabatic_cst[vi];
    Real volume_ratio = in_volume[vi] / in_old_volume[vi];
    Real x = 0.5 * (adiabatic_cst - 1.0);
    Real numer_accrois_nrj = 1.0 + x * (1.0 - volume_ratio);
    Real denom_accrois_nrj = 1.0 + x * (1.0 - (1.0 / volume_ratio));
    Real internal_energy = in_out_internal_energy[vi];
    internal_energy = internal_energy * (numer_accrois_nrj / denom_accrois_nrj);

    // Prise en compte du travail des forces de viscosité
    if (add_viscosity_force)
      internal_energy = internal_energy - deltatf * in_viscosity_work[vi] / (in_cell_mass[vi] * denom_accrois_nrj);

    in_out_internal_energy[vi] = internal_energy;

    Real density = in_density[vi];
    Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
    Real sound_speed = math::sqrt(adiabatic_cst * pressure / density);
    out_pressure[vi] = pressure;
    out_sound_speed[vi] = sound_speed;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des nouveaux pas de temps.
 */
void MicroHydroModule::
computeDeltaT()
{
  const Real old_dt = m_global_deltat();

  // Calcul du pas de temps pour le respect du critère de CFL

  Real minimum_aux = FloatInfo<Real>::maxValue();

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    ax::ReducerMin<double> minimum_aux_reducer(command);
    auto in_sound_speed = viewIn(command, m_sound_speed);
    auto in_caracteristic_length = viewIn(command, m_caracteristic_length);
    command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells())
    {
      Real cell_dx = in_caracteristic_length[cid];
      Real sound_speed = in_sound_speed[cid];
      Real dx_sound = cell_dx / sound_speed;
      minimum_aux_reducer.min(dx_sound);
    };
    minimum_aux = minimum_aux_reducer.reduce();
  }

  Real new_dt = options()->getCfl() * minimum_aux;

  // Pas de variations trop brutales à la hausse comme à la baisse
  Real max_dt = (ARCANE_REAL(1.0) + options()->getVariationSup()) * old_dt;
  Real min_dt = (ARCANE_REAL(1.0) - options()->getVariationInf()) * old_dt;

  new_dt = math::min(new_dt, max_dt);
  new_dt = math::max(new_dt, min_dt);

  Real max_density_ratio = m_density_ratio_maximum();

  // control de l'accroissement relatif de la densité
  Real dgr = options()->getDensityGlobalRatio();
  if (max_density_ratio > dgr)
    new_dt = math::min(old_dt * dgr / max_density_ratio, new_dt);

  IParallelMng* pm = mesh()->parallelMng();
  new_dt = pm->reduce(Parallel::ReduceMin, new_dt);

  // Respect des valeurs min et max imposées par le fichier de données .plt
  new_dt = math::min(new_dt, options()->getDeltatMax());
  new_dt = math::max(new_dt, options()->getDeltatMin());

  if (options()->getCheckNumericalResult()) {
    Integer iteration = m_global_iteration();
    if (iteration < 25) {
      Real ref_new_dt = reference_global_deltat[iteration];
      if (!math::isNearlyEqual(new_dt, ref_new_dt))
        ARCANE_FATAL("Bad value for 'new_dt' ref={0} v={1} diff={2}",
                     ref_new_dt, new_dt, (new_dt - ref_new_dt) / ref_new_dt);
    }
  }

  // Le dernier calcul se fait exactement au temps stopTime()
  {
    Real stop_time = options()->getFinalTime();
    bool not_yet_finish = (m_global_time() < stop_time);
    bool too_much = ((m_global_time() + new_dt) > stop_time);

    if (not_yet_finish && too_much) {
      new_dt = stop_time - m_global_time();
      subDomain()->timeLoopMng()->stopComputeLoop(true);
    }
  }

  // Mise à jour des variables
  m_old_dt_f.assign(old_dt);
  m_delta_t_n.assign(ARCANE_REAL(0.5) * (old_dt + new_dt));
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
inline void MicroHydroModule::
computeCQs(Real3 node_coord[8], Real3 face_coord[6], Span<Real3> cqs)
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
  const Real3 n1a04 = demi * math::cross(node_coord[0] - c0, node_coord[3] - c0);
  const Real3 n1a03 = demi * math::cross(node_coord[3] - c0, node_coord[2] - c0);
  const Real3 n1a02 = demi * math::cross(node_coord[2] - c0, node_coord[1] - c0);
  const Real3 n1a01 = demi * math::cross(node_coord[1] - c0, node_coord[0] - c0);

  // Calcul des normales face 2 :
  const Real3 n2a05 = demi * math::cross(node_coord[0] - c1, node_coord[4] - c1);
  const Real3 n2a12 = demi * math::cross(node_coord[4] - c1, node_coord[7] - c1);
  const Real3 n2a08 = demi * math::cross(node_coord[7] - c1, node_coord[3] - c1);
  const Real3 n2a04 = demi * math::cross(node_coord[3] - c1, node_coord[0] - c1);

  // Calcul des normales face 3 :
  const Real3 n3a01 = demi * math::cross(node_coord[0] - c2, node_coord[1] - c2);
  const Real3 n3a06 = demi * math::cross(node_coord[1] - c2, node_coord[5] - c2);
  const Real3 n3a09 = demi * math::cross(node_coord[5] - c2, node_coord[4] - c2);
  const Real3 n3a05 = demi * math::cross(node_coord[4] - c2, node_coord[0] - c2);

  // Calcul des normales face 4 :
  const Real3 n4a09 = demi * math::cross(node_coord[4] - c3, node_coord[5] - c3);
  const Real3 n4a10 = demi * math::cross(node_coord[5] - c3, node_coord[6] - c3);
  const Real3 n4a11 = demi * math::cross(node_coord[6] - c3, node_coord[7] - c3);
  const Real3 n4a12 = demi * math::cross(node_coord[7] - c3, node_coord[4] - c3);

  // Calcul des normales face 5 :
  const Real3 n5a02 = demi * math::cross(node_coord[1] - c4, node_coord[2] - c4);
  const Real3 n5a07 = demi * math::cross(node_coord[2] - c4, node_coord[6] - c4);
  const Real3 n5a10 = demi * math::cross(node_coord[6] - c4, node_coord[5] - c4);
  const Real3 n5a06 = demi * math::cross(node_coord[5] - c4, node_coord[1] - c4);

  // Calcul des normales face 6 :
  const Real3 n6a03 = demi * math::cross(node_coord[2] - c5, node_coord[3] - c5);
  const Real3 n6a08 = demi * math::cross(node_coord[3] - c5, node_coord[7] - c5);
  const Real3 n6a11 = demi * math::cross(node_coord[7] - c5, node_coord[6] - c5);
  const Real3 n6a07 = demi * math::cross(node_coord[6] - c5, node_coord[2] - c5);

  const Real real_1div12 = ARCANE_REAL(1.0) / ARCANE_REAL(12.0);

  // Calcul des résultantes aux sommets :
  cqs[0] = (five * (n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
            (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09)) *
  real_1div12;
  cqs[1] = (five * (n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
            (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07)) *
  real_1div12;
  cqs[2] = (five * (n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
            (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08)) *
  real_1div12;
  cqs[3] = (five * (n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
            (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11)) *
  real_1div12;
  cqs[4] = (five * (n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
            (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11)) *
  real_1div12;
  cqs[5] = (five * (n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
            (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02)) *
  real_1div12;
  cqs[6] = (five * (n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
            (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08)) *
  real_1div12;
  cqs[7] = (five * (n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
            (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03)) *
  real_1div12;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du volume des mailles, des longueurs caractéristiques
 * et des résultantes aux sommets.
 */
void MicroHydroModule::
computeGeometricValues()
{
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  auto in_node_coord = viewIn(command, m_node_coord);
  auto in_out_cell_cqs = viewInOut(command, m_cell_cqs);
  auto in_volume = viewIn(command, m_volume);

  auto out_volume = viewOut(command, m_volume);
  auto out_old_volume = viewOut(command, m_old_volume);
  auto out_caracteristic_length = viewOut(command, m_caracteristic_length);

  auto cnc = m_connectivity_view.cellNode();

  command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells())
  {
    auto nodes = cnc.nodes(cid);

    // Copie locale des coordonnées des sommets d'une maille
    Real3 coord[8] = {
      in_node_coord[nodes[0]], in_node_coord[nodes[1]],
      in_node_coord[nodes[2]], in_node_coord[nodes[3]],
      in_node_coord[nodes[4]], in_node_coord[nodes[5]],
      in_node_coord[nodes[6]], in_node_coord[nodes[7]]
    };

    // Coordonnées des centres des faces
    Real3 face_coord[6] = {
      0.25 * (coord[0] + coord[3] + coord[2] + coord[1]),
      0.25 * (coord[0] + coord[4] + coord[7] + coord[3]),
      0.25 * (coord[0] + coord[1] + coord[5] + coord[4]),
      0.25 * (coord[4] + coord[5] + coord[6] + coord[7]),
      0.25 * (coord[1] + coord[2] + coord[6] + coord[5]),
      0.25 * (coord[2] + coord[3] + coord[7] + coord[6]),
    };

    // Calcule la longueur caractéristique de la maille.
    {
      Real3 median1 = face_coord[0] - face_coord[3];
      Real3 median2 = face_coord[2] - face_coord[5];
      Real3 median3 = face_coord[1] - face_coord[4];
      Real d1 = median1.normL2();
      Real d2 = median2.normL2();
      Real d3 = median3.normL2();

      Real dx_numerator = d1 * d2 * d3;
      Real dx_denominator = d1 * d2 + d1 * d3 + d2 * d3;
      out_caracteristic_length[cid] = dx_numerator / dx_denominator;
    }

    // Calcule les résultantes aux sommets
    computeCQs(coord, face_coord, in_out_cell_cqs[cid]);

    Span<const Real3> in_cqs(in_out_cell_cqs[cid]);

    // Calcule le volume de la maille
    {
      Real volume = 0.0;
      for (Integer i_node = 0; i_node < 8; ++i_node)
        volume += math::dot(coord[i_node], in_cqs[i_node]);
      volume /= 3.0;

      out_old_volume[cid] = in_volume[cid];
      out_volume[cid] = volume;
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
hydroInit()
{
  info() << "INIT: DTmin=" << options()->getDeltatMin()
         << " DTmax=" << options()->getDeltatMax()
         << " DT=" << m_global_deltat();
  if (m_global_deltat() > options()->getDeltatMax())
    ARCANE_FATAL("DeltaT > DTMax");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
_computeNodeIndexInCells()
{
  info() << "ComputeNodeIndexInCells";
  // Un noeud est connecté au maximum à MAX_NODE_CELL mailles
  // Calcul pour chaque noeud son index dans chacune des
  // mailles à laquelle il est connecté.
  NodeGroup nodes = allNodes();
  Integer nb_node = nodes.size();
  m_node_index_in_cells.resize(MAX_NODE_CELL * nb_node);
  m_node_index_in_cells.fill(-1);
  auto node_cell_cty = m_connectivity_view.nodeCell();
  auto cell_node_cty = m_connectivity_view.cellNode();
  ENUMERATE_NODE (inode, nodes) {
    NodeLocalId node = *inode;
    Int32 index = 0;
    Int32 first_pos = node.localId() * MAX_NODE_CELL;
    for (CellLocalId cell : node_cell_cty.cells(node)) {
      Int16 node_index_in_cell = 0;
      for (NodeLocalId cell_node : cell_node_cty.nodes(cell)) {
        if (cell_node == node)
          break;
        ++node_index_in_cell;
      }
      m_node_index_in_cells[first_pos + index] = node_index_in_cell;
      ++index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
hydroExit()
{
  info() << "Hydro exit entry point";
  m_time_stats->dumpCurrentStats("SH_DoOneIteration");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
_doCall(const char* func_name, std::function<void()> func)
{
  {
    Timer::Sentry ts_elapsed(&m_elapsed_timer);
    Timer::Action ts_action1(m_time_stats, func_name);
    func();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define DO_CALL(func_name) \
  _doCall(#func_name, [=] { this->func_name(); })

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
doOneIteration()
{
  DO_CALL(computeForces);
  DO_CALL(computeVelocity);
  DO_CALL(computeViscosityWork);
  DO_CALL(applyBoundaryCondition);
  DO_CALL(moveNodes);
  DO_CALL(computeGeometricValues);
  DO_CALL(updateDensity);
  DO_CALL(applyEquationOfState);
  DO_CALL(computeDeltaT);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(MicroHydroModule, MicroHydro);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MicroHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
