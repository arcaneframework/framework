// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

/**
 * \author Pascal Have
 * \version 1.0
 * \brief Interface des systemes lineaires. L'interface \class IIndexManager permet
 * de construire le graphe du syt�me lin�aire (entr�e non nuls)
 *
 * Le syst�me peut �tre construit en une ou deux passes .
 * Une passe permet de d�finir les entr�es non nulles du graphe.
 * \fn prepare permet d'achever cette passe
 * La seconde permet de donner les valeurs du syst�me � l'aide
 * des fonction \fn addData, \fn setData , \fn addRHSData \fn setRHSData .
 * Elle s'ach�ve par la m�thode \fn finalize
 *
 * Exemple d'utilisation :
 * \code
 * ILinearSolver* solver = options()->linearSolver() ;
 * ILinearSystem* system = solver->getLinearSystem() ;
 * IIndexManager * manager = system->getManager();
 * const IIndexManager::Entry pressure_entry = manager->buildVariableEntry(m_pressure.variable(),IIndexManager::Direct);
 * const IIndexManager::Equation water_equation = manager->buildEquation("Water",water_entry);
 * ENUMERATE_CELL(icell,allCells())
 * {
 *    const Cell& cell = *icell ;
 *    const IIndexManager::EquationIndex equation_index
          = manager->defineEquationIndex(waterEquation,cell);
 *    const IIndexManager::EntryIndex pressure_index = manager->defineEntryIndex(pressureEntry, cell );
 *    system->defineData(equation_index,pressure_index)
 * }
 * system->prepare() ;
 * ENUMERATE_CELL(icell,allCells())
 * {
 *    const Cell& cell = *icell ;
 *    Real diag_value = function(cell) ;
 *    const IIndexManager::EquationIndex equation_index = manager->getEquationIndex(water_equation,cell);
 *    const IIndexManager::EntryIndex diag_entry_index = manager->getEntryIndex(pressure_entry, cell );
 *    system->addData(equation_index,sign,diag_entry_index,diag_value) ;
 *  }
 * system->finalize() ;
 *
 * \todo Il faut �purer l'interface pour distinguer le syst�me lin�aire produit
 * de sa construction et ainsi permettre d'avoir des impl�mentations sous les formes
 * ThreeStep (definition + localisation + remplissage), OneStep (remplissage avec d�finition implicite),
 * Working (permet des retouches sur la matrice construite ou en cours de construction)
 * Interface du service du modu?le de resolution non lineaire.
 */


namespace Alien {

class ILinearSystemVisitor;
class ILinearSystemBuilder;

class ILinearSystem
{
public:
  /** Constructeur de la classe */
  ILinearSystem()
    {
      ;
    }

  /** Destructeur de la classe */
  virtual ~ILinearSystem() { }

public:

  //! Initialisation
  virtual void init() = 0;

  //! Adding operators to interface
  virtual bool connect(ILinearSystemBuilder * builder)
  { throw Arccore::FatalErrorException(A_FUNCINFO,"not implemented"); }

  virtual bool accept(ILinearSystemVisitor * visitor) = 0 ;

  //! Initializing resolution step
  virtual void start() = 0 ;

  //! Finalizing resolution step et free temporary object
  virtual void end() = 0 ;

  //! Getting name
  virtual const char * name() const = 0;
};
}

