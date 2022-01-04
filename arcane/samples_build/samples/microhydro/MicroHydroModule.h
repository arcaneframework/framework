// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef MICROHYDROMODULE_H
#define MICROHYDROMODULE_H

#include "TypesMicroHydro.h"

#include <arcane/geometry/IGeometryMng.h>

#include "MicroHydro_axl.h"

using namespace Arcane;

/**
 * Représente un module d'hydrodynamique lagrangienne très simplifié :
 *   - le seul type de maille supporté est l'hexaèdre,
 *   - pas de pseudo viscosité supportée,
 *   - le seul type de calcul de longueur caractéristique supporté est celui utilisant les médianes,
 *   - le seul type de condition aux limites supporté est d'imposer une composante de la vitesse sur une surface,
 *   - la masse nodale est supposée constante et n'est pas recalculée à chaque itération,
 *   - aucun test de cohérence des valeurs (pression positive, volume positif, ...)  n'est effectué.
 *  
 * La liste des opérations effectuées par le module est la suivante :
 *   - calcul des forces de pression,
 *   - calcul de l'impulsion,
 *   - prise en compte des conditions aux limites,
 *   - déplacement des noeuds,
 *   - calcul des nouvelles valeurs géométriques : volume des mailles, longueur caractéristique des mailles,
 *     resultantes aux sommets de chaque maille,
 *   - calcul de la densité,
 *   - calcul de la pression et de l'énergie par l'équation d'état. Ce calcul est effectué par un service
 *     ARCANE. Deux implémentations sont disponibles pour le service : gaz parfait, et "stiffened" gaz.
 *   - calcul du nouveau pas de temps.
 * 
 */
class MicroHydroModule
: public ArcaneMicroHydroObject
{
 public:
  /** Constructeur de la classe */
  MicroHydroModule(const ModuleBuildInfo& mbi)
    : ArcaneMicroHydroObject(mbi) {}
  /** Destructeur de la classe */
  ~MicroHydroModule() {}
  
 public:
  /** 
   *  Initialise le module. 
   *  L'initialisation comporte deux parties distinctes:
   *  - la première partie où il faut indiquer la taille des variables
   *    tableaux. Dans notre cas, il s'agit de \c m_cell_cqs et
   *    \c m_viscosity_force, qui sont toutes deux des variables
   *    aux mailles possédant une valeur pour chaque noeud de chaque
   *    maille. Comme on ne supporte que les héxaèdres, il y a 8 valeurs 
   *    par maille,
   *  - la deuxième partie qui consiste à initialiser les variables avec
   *    leur valeur de départ. Pour les variables \c Pressure, \c Density et
   *    \c AdiabaticCst, c'est ARCANE qui les initialisent directement
   *    à partir du jeu de donnée. La variable \c NodeCoord est aussi
   *    initialisée par l'architecture lors de la lecture du maillage. Les
   *    autres variables sont calculées comme suit :
   *    - le pas de temps initial est donné par le jeu de donnée,
   *    - les valeurs géométriques (longueur caractéristique, volume et
   *      résultantes aux sommets) sont calculées à partir des coordonnées des
   *      noeuds,
   *    - la masse des mailles est calculée à partir de sa densité et de
   *      son volume,
   *    - la masse des mailles et la masse nodale. La masse d'une maille
   *      est calculée à partir de sa densité et de son volume,
   *    - la masse nodale est calculée en ajoutant les contributions de
   *      chaque maille connecté à un noeud donné. Chaque maille
   *      contribue pour 1/8ème de sa masse à la masse nodale de chacun de ses
   *      sommets,
   *    - l'énergie interne et la vitesse du son sont calculées en fonction 
   *      de l' équation d'état.
   */
  virtual void hydroStartInit();
  
  /** 
   *  Initialise le module en cas de reprise. 
   *  Afin d'éviter de sauvegarder le volume des mailles, cette méthode
   *  recalcule le volume en fonction des coordonnées.
   */
  virtual void hydroContinueInit();
  
  /** 
   * Calcule la contribution des forces de pression par
   * noeud au temps courant \f$t^{n}\f$. Pour chaque noeud de chaque maille, 
   * il s'agit de la pression multipliée par la résultante en ce noeud.
   * Calcule les forces de pression au temps courant \f$t^{n}\f$.
   */
  virtual void computePressureForce();
		
  /**
   * Calcule la force (\c m_force) qui s'applique aux noeuds en
   * ajoutant l'éventuelle contribution de la pseudo-viscosité. Calcule 
   * ensuite la nouvelle vitesse (\c m_velocity) aux noeuds. 
   */
  virtual void computeVelocity();
		
  /**
   * Applique les conditions aux limites.
   * Les conditions aux limites dépendent des options du
   * jeu de données. Dans cette implémentation, une condition aux limites
   * possède les propriétés suivantes :
   * - un type: trois types sont supportés: contraindre la composante
   * \f$x\f$ du vecteur vitesse, contraindre la composante \f$y\f$ du vecteur
   * vitesse ou contraindre la composante \f$z\f$ du vecteur vitesse,
   * - une valeur: il s'agit d'un réel indiquant la valeur de la
   * contrainte,
   * - une surface: il s'agit de la surface sur laquelle s'applique la
   * contrainte.
   * 
   * Appliquer les conditions aux limites consiste donc à fixer une
   * composante d'un vecteur vitesse pour chaque noeud de chaque face de
   * chaque surface sur laquelle on impose une condition aux limites.
   */		
  virtual void applyBoundaryCondition();
		
  /**
   * Modifie les coordonnées (\c m_node_coord)
   * des noeuds d'après la valeur du vecteur vitesse et du pas de temps.
   */
  virtual void moveNodes();
		
  /**
   * Ce point d'entrée regroupe l'ensemble des calculs géométriques
   * utiles pour le schéma. Dans notre cas, il s'agit pour chaque maille :
   * - de calculer sa longueur caractéristique,
   * - de calculer les résultantes à ses sommets,
   * - de calculer son volume.
   
   * Pour optimiser le calcul (utilisation du cache), à chaque itération 
   * sur une maille, sont stockées localement les coordonnées de ses noeuds 
   * et celles du centre de ses faces.
   */
  virtual void computeGeometricValues();
  
  /**
   * Calcule la nouvelle valeur de la densité des
   * mailles, en considérant que la masse d'une maille est constante au
   * cours du temps. Dans ce cas, la nouvelle densité est égale à la masse
   * divisée par le nouveau volume.
   */
  virtual void updateDensity();
		
  /**
   * Ce point d'entrée calcule l'énergie interne, la pression et la vitesse
   * du son dans la maille en faisant appel au service d'équation d'état.
   */
  virtual void applyEquationOfState();
		
  /**
   * Détermine la valeur du pas de temps pour l'itération suivante. 
   * Le pas de temps est contraint par :
   * - la valeur de la CFL,
   * - les valeurs \c deltatMin() et \c deltatMax() du jeu de données,
   * - la valeur du temps final. Lors de la dernière itération, le pas
   *   de temps doit être tel qu'on s'arrête exactement au temps spécifié
   *   dans le jeu de données (\c finalTime()).
   */
  virtual void computeDeltaT();

  /** Retourne le numéro de version du module */
  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,0); }
  
 private:
  /**
   * Calcule les résultantes aux noeuds d'une maille hexaédrique.
   * La méthode utilisée est celle du découpage en quatre triangles.
   * Méthode appelée par le point d'entrée \c computeGeometricValues()
   */
  inline void computeCQs(Real3 node_coord[8],Real3 face_coord[6],const Cell& cell);
};

#endif
