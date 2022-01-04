// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "MicroHydroModule.h"

#include <arcane/MathUtils.h>
#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>

#include <arcane/geometry/IGeometry.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
hydroStartInit()
{
  // Dimensionne les variables tableaux
  m_cell_cqs.resize(8);

    // Initialise le delta-t
  Real deltat_init = options()->deltatInit();
  m_global_deltat = deltat_init;

  // Initialise les données géométriques: volume, cqs, longueurs caractéristiques
  computeGeometricValues();

  // Initialisation de la masses des mailles et des masses nodale
  ENUMERATE_CELL(icell, allCells()){
    Cell cell = * icell;
    m_cell_mass[icell] = m_density[icell] * m_cell_volume[icell];

    Real contrib_node_mass = 0.125 * m_cell_mass[cell];
    for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode){
      m_node_mass[inode] += contrib_node_mass;
    }
  }

  m_node_mass.synchronize();

  // Initialise l'énergie et la vitesse du son
  options()->eosModel()->initEOS(allCells());

  Arcane::Numerics::IGeometryMng* gm = options()->geometry();
  gm->init();
  auto g = gm->geometry();
  bool is_verbose = false;
  if (is_verbose){
    ENUMERATE_CELL(icell,allCells()){
      Cell c = *icell;
      info() << "Volume = " << g->computeMeasure(c);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
hydroContinueInit()
{
  ENUMERATE_CELL(icell, allCells()){
    Cell cell = *icell;

    // Calcule le volume de la maille
    Real volume = 0.0;
    for (Integer inode = 0; inode < 8; ++inode) 
      volume += math::dot(m_node_coord[cell.node(inode)], m_cell_cqs[icell] [inode]);
    volume /= 3.;
    
    m_cell_volume[icell] = volume;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
computePressureForce()
{
  // Remise à zéro du vecteur des forces.
  m_force.fill(Real3::null());

  // Calcul pour chaque noeud de chaque maille la contribution
  // des forces de pression
  ENUMERATE_CELL(icell, allCells()){
    Cell cell = * icell;
    Real pressure = m_pressure[icell];
    for (NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode)
      m_force[inode] += pressure * m_cell_cqs[icell] [inode.index()];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
computeVelocity()
{
  // Calcule l'impulsion aux noeuds
  ENUMERATE_NODE(inode, ownNodes()){
    Real node_mass = m_node_mass[inode];

    Real3 old_velocity = m_velocity[inode];
    Real3 new_velocity = old_velocity + (m_global_deltat() / node_mass) * m_force[inode];

    m_velocity[inode] = new_velocity;
  }

  m_velocity.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
applyBoundaryCondition()
{
  for (Integer i = 0, nb = options()->boundaryCondition.size(); i < nb; ++i){
    FaceGroup face_group = options()->boundaryCondition[i]->surface();
    Real value = options()->boundaryCondition[i]->value();
    TypesMicroHydro::eBoundaryCondition type = options()->boundaryCondition[i]->type();

    // boucle sur les faces de la surface
    ENUMERATE_FACE(j, face_group){
      Face face = * j;
      Integer nb_node = face.nbNode();

      // boucle sur les noeuds de la face
      for (Integer k = 0; k < nb_node; ++k){
        Node node = face.node(k);
        Real3& velocity = m_velocity[node];

        switch (type){
        case TypesMicroHydro::VelocityX:
          velocity.x = value;
          break;
        case TypesMicroHydro::VelocityY:
          velocity.y = value;
          break;
        case TypesMicroHydro::VelocityZ:
          velocity.z = value;
          break;
        case TypesMicroHydro::Unknown:
          break;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
moveNodes()
{
  Real deltat = m_global_deltat();

  ENUMERATE_NODE(inode, allNodes()){
    m_node_coord[inode] += deltat * m_velocity[inode];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
computeGeometricValues()
{
  m_old_cell_volume.copy(m_cell_volume);

  // Copie locale des coordonnées des sommets d'une maille
  Real3 coord[8];
  // Coordonnées des centres des faces
  Real3 face_coord[6];

  ENUMERATE_CELL(icell, allCells()){
    Cell cell = * icell;

    // Recopie les coordonnées locales (pour le cache)
    for (NodeEnumerator inode(cell.nodes()); inode.index() < 8; ++inode) 
      coord[inode.index()] = m_node_coord[inode];

    // Calcul les coordonnées des centres des faces
    face_coord[0] = 0.25 * (coord[0] + coord[3] + coord[2] + coord[1]);
    face_coord[1] = 0.25 * (coord[0] + coord[4] + coord[7] + coord[3]);
    face_coord[2] = 0.25 * (coord[0] + coord[1] + coord[5] + coord[4]);
    face_coord[3] = 0.25 * (coord[4] + coord[5] + coord[6] + coord[7]);
    face_coord[4] = 0.25 * (coord[1] + coord[2] + coord[6] + coord[5]);
    face_coord[5] = 0.25 * (coord[2] + coord[3] + coord[7] + coord[6]);

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
      m_caracteristic_length[icell] = dx_numerator / dx_denominator;
    }

    // Calcule les résultantes aux sommets
    computeCQs(coord, face_coord, cell);

    // Calcule le volume de la maille
    {
      Real volume = 0.;
      for (Integer inode = 0; inode < 8; ++inode) 
        volume += math::scaMul(coord[inode], m_cell_cqs[icell] [inode]);
      volume /= 3.;

      m_cell_volume[icell] = volume;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
updateDensity()
{
  ENUMERATE_CELL(icell,ownCells()){
    // Real old_density = m_density[icell];
    Real new_density = m_cell_mass[icell] / m_cell_volume[icell];

    m_density[icell] = new_density;
  }

  m_density.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
applyEquationOfState()
{
  // Calcul de l'énergie interne
  ENUMERATE_CELL(icell, allCells()){
    Real adiabatic_cst = m_adiabatic_cst[icell];
    Real volume_ratio = m_cell_volume[icell] / m_old_cell_volume[icell];
    Real x = 0.5 * (adiabatic_cst - 1.);
    Real numer_accrois_nrj = 1. + x * (1. - volume_ratio);
    Real denom_accrois_nrj = 1. + x * (1. - 1. / volume_ratio);
    //denom_accrois_nrj = 0.0;
    m_internal_energy[icell] *= numer_accrois_nrj / denom_accrois_nrj;
  }

  // Calcul de la pression et de la vitesse du son
  options()->eosModel()->applyEOS(allCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::
computeDeltaT()
{
  // const Real old_dt = m_global_deltat();

  // Calcul du pas de temps pour le respect du critère de CFL

  Real minimum_aux = FloatInfo < Real >::maxValue();
  Real new_dt = FloatInfo < Real >::maxValue();

  ENUMERATE_CELL(icell, ownCells()){
    Real cell_dx = m_caracteristic_length[icell];
    // Real density = m_density[icell];
    // Real pressure = m_pressure[icell];
    Real sound_speed = m_sound_speed[icell];
    Real dx_sound = cell_dx / sound_speed;
    minimum_aux = math::min(minimum_aux, dx_sound);
  }

  new_dt = options()->cfl() * minimum_aux;

  new_dt = parallelMng()->reduce(Parallel::ReduceMin, new_dt);

  // respect des valeurs min et max imposées par le fichier de données .plt
  new_dt = math::min(new_dt, options()->deltatMax());
  new_dt = math::max(new_dt, options()->deltatMin());

  // Real data_min_max_dt = new_dt;

  // Le dernier calcul se fait exactement au temps stopTime()
  Real stop_time  = options()->finalTime();
  bool not_yet_finish = (m_global_time() < stop_time);
  bool too_much = ((m_global_time()+new_dt) > stop_time);

  if ( not_yet_finish && too_much ){
    new_dt = stop_time - m_global_time();
    subDomain()->timeLoopMng()->stopComputeLoop(true);
  }

  // Mise à jour du pas de temps
  m_global_deltat = new_dt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void MicroHydroModule::
computeCQs(Real3 node_coord[8], Real3 face_coord[6], const Cell & cell)
{
  const Real3 c0 = face_coord[0];
  const Real3 c1 = face_coord[1];
  const Real3 c2 = face_coord[2];
  const Real3 c3 = face_coord[3];
  const Real3 c4 = face_coord[4];
  const Real3 c5 = face_coord[5];

  // Calcul des normales face 1 :
  const Real3 n1a04 = 0.5 * math::vecMul(node_coord[0] - c0, node_coord[3] - c0);
  const Real3 n1a03 = 0.5 * math::vecMul(node_coord[3] - c0, node_coord[2] - c0);
  const Real3 n1a02 = 0.5 * math::vecMul(node_coord[2] - c0, node_coord[1] - c0);
  const Real3 n1a01 = 0.5 * math::vecMul(node_coord[1] - c0, node_coord[0] - c0);

  // Calcul des normales face 2 :
  const Real3 n2a05 = 0.5 * math::vecMul(node_coord[0] - c1, node_coord[4] - c1);
  const Real3 n2a12 = 0.5 * math::vecMul(node_coord[4] - c1, node_coord[7] - c1);
  const Real3 n2a08 = 0.5 * math::vecMul(node_coord[7] - c1, node_coord[3] - c1);
  const Real3 n2a04 = 0.5 * math::vecMul(node_coord[3] - c1, node_coord[0] - c1);

  // Calcul des normales face 3 :
  const Real3 n3a01 = 0.5 * math::vecMul(node_coord[0] - c2, node_coord[1] - c2);
  const Real3 n3a06 = 0.5 * math::vecMul(node_coord[1] - c2, node_coord[5] - c2);
  const Real3 n3a09 = 0.5 * math::vecMul(node_coord[5] - c2, node_coord[4] - c2);
  const Real3 n3a05 = 0.5 * math::vecMul(node_coord[4] - c2, node_coord[0] - c2);

  // Calcul des normales face 4 :
  const Real3 n4a09 = 0.5 * math::vecMul(node_coord[4] - c3, node_coord[5] - c3);
  const Real3 n4a10 = 0.5 * math::vecMul(node_coord[5] - c3, node_coord[6] - c3);
  const Real3 n4a11 = 0.5 * math::vecMul(node_coord[6] - c3, node_coord[7] - c3);
  const Real3 n4a12 = 0.5 * math::vecMul(node_coord[7] - c3, node_coord[4] - c3);

  // Calcul des normales face 5 :
  const Real3 n5a02 = 0.5 * math::vecMul(node_coord[1] - c4, node_coord[2] - c4);
  const Real3 n5a07 = 0.5 * math::vecMul(node_coord[2] - c4, node_coord[6] - c4);
  const Real3 n5a10 = 0.5 * math::vecMul(node_coord[6] - c4, node_coord[5] - c4);
  const Real3 n5a06 = 0.5 * math::vecMul(node_coord[5] - c4, node_coord[1] - c4);

  // Calcul des normales face 6 :
  const Real3 n6a03 = 0.5 * math::vecMul(node_coord[2] - c5, node_coord[3] - c5);
  const Real3 n6a08 = 0.5 * math::vecMul(node_coord[3] - c5, node_coord[7] - c5);
  const Real3 n6a11 = 0.5 * math::vecMul(node_coord[7] - c5, node_coord[6] - c5);
  const Real3 n6a07 = 0.5 * math::vecMul(node_coord[6] - c5, node_coord[2] - c5);

  // Calcul des résultantes aux sommets :
  m_cell_cqs[cell] [0] = (5. * (n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
                          (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09)) * (1. / 12.);
  m_cell_cqs[cell] [1] = (5. * (n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
                          (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07)) * (1. / 12.);
  m_cell_cqs[cell] [2] = (5. * (n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
                          (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08)) * (1. / 12.);
  m_cell_cqs[cell] [3] = (5. * (n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
                          (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11)) * (1. / 12.);
  m_cell_cqs[cell] [4] = (5. * (n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
                          (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11)) * (1. / 12.);
  m_cell_cqs[cell] [5] = (5. * (n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
                          (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02)) * (1. / 12.);
  m_cell_cqs[cell] [6] = (5. * (n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
                          (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08)) * (1. / 12.);
  m_cell_cqs[cell] [7] = (5. * (n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
                          (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03)) * (1. / 12.);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MICROHYDRO(MicroHydroModule);
